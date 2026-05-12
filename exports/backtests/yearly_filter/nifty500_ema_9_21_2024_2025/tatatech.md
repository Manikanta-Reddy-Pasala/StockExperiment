# Tata Technologies Ltd. (TATATECH)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-11 15:15:00 (3716 bars)
- **Last close:** 630.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 157 |
| ALERT1 | 106 |
| ALERT2 | 106 |
| ALERT2_SKIP | 50 |
| ALERT3 | 301 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 98 |
| PARTIAL | 16 |
| TARGET_HIT | 8 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 43 / 72
- **Target hits / Stop hits / Partials:** 8 / 91 / 16
- **Avg / median % per leg:** 1.13% / -0.56%
- **Sum % (uncompounded):** 129.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 4 | 16.0% | 1 | 24 | 0 | -0.03% | -0.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 4 | 16.0% | 1 | 24 | 0 | -0.03% | -0.6% |
| SELL (all) | 90 | 39 | 43.3% | 7 | 67 | 16 | 1.45% | 130.2% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.73% | 8.2% |
| SELL @ 3rd Alert (retest2) | 87 | 37 | 42.5% | 7 | 65 | 15 | 1.40% | 122.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.73% | 8.2% |
| retest2 (combined) | 112 | 41 | 36.6% | 8 | 89 | 15 | 1.08% | 121.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1041.70 | 1025.06 | 1022.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 1047.25 | 1029.50 | 1025.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1050.90 | 1052.25 | 1043.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 1050.90 | 1052.25 | 1043.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1048.65 | 1051.88 | 1047.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 1048.25 | 1051.88 | 1047.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1050.05 | 1050.91 | 1047.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 1047.10 | 1050.91 | 1047.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1045.60 | 1049.85 | 1047.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 1047.85 | 1049.85 | 1047.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1047.70 | 1049.42 | 1047.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:30:00 | 1045.50 | 1049.42 | 1047.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 1044.80 | 1048.50 | 1047.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:00:00 | 1044.80 | 1048.50 | 1047.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 1045.30 | 1047.86 | 1047.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:45:00 | 1044.15 | 1047.86 | 1047.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 1046.00 | 1046.86 | 1046.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 1052.00 | 1046.86 | 1046.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 13:15:00 | 1049.80 | 1052.07 | 1052.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 13:15:00 | 1049.80 | 1052.07 | 1052.11 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 1055.95 | 1052.33 | 1052.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 1068.10 | 1057.03 | 1054.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 1074.15 | 1086.56 | 1078.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 1074.15 | 1086.56 | 1078.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1074.15 | 1086.56 | 1078.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 1077.35 | 1086.56 | 1078.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1072.55 | 1083.75 | 1077.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:30:00 | 1071.30 | 1083.75 | 1077.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 1066.85 | 1074.07 | 1074.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 1062.55 | 1070.46 | 1072.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 09:15:00 | 1060.80 | 1059.41 | 1064.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1060.80 | 1059.41 | 1064.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1060.80 | 1059.41 | 1064.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 1063.90 | 1059.41 | 1064.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1046.00 | 1043.38 | 1050.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 1038.85 | 1042.70 | 1049.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:00:00 | 1039.15 | 1041.99 | 1048.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 986.91 | 1021.21 | 1033.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 987.19 | 1021.21 | 1033.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 10:15:00 | 1020.50 | 1018.42 | 1027.20 | SL hit (close>ema200) qty=0.50 sl=1018.42 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1052.90 | 1030.75 | 1029.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1066.30 | 1048.57 | 1040.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 1056.10 | 1058.57 | 1052.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 1056.10 | 1058.57 | 1052.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1057.55 | 1060.06 | 1058.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 1056.50 | 1060.06 | 1058.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1057.45 | 1059.53 | 1058.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 1057.45 | 1059.53 | 1058.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1059.20 | 1059.47 | 1058.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:30:00 | 1056.00 | 1059.47 | 1058.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1058.55 | 1059.28 | 1058.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 1058.55 | 1059.28 | 1058.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1060.60 | 1059.55 | 1058.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 1059.60 | 1059.55 | 1058.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1058.45 | 1059.33 | 1058.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:30:00 | 1057.60 | 1059.33 | 1058.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1054.85 | 1058.43 | 1058.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:45:00 | 1054.65 | 1058.43 | 1058.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 1052.75 | 1057.30 | 1057.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 12:15:00 | 1049.00 | 1055.64 | 1056.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 15:15:00 | 1029.00 | 1028.19 | 1033.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 15:15:00 | 1029.00 | 1028.19 | 1033.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1029.00 | 1028.19 | 1033.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 1010.00 | 1028.19 | 1033.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1008.50 | 1005.06 | 1005.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 1008.50 | 1005.06 | 1005.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 10:15:00 | 1019.70 | 1007.99 | 1006.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 11:15:00 | 1017.55 | 1020.37 | 1017.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 11:15:00 | 1017.55 | 1020.37 | 1017.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 1017.55 | 1020.37 | 1017.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 1017.55 | 1020.37 | 1017.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 1017.65 | 1019.83 | 1017.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:30:00 | 1016.40 | 1019.83 | 1017.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 1014.20 | 1018.70 | 1016.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:00:00 | 1014.20 | 1018.70 | 1016.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 1014.80 | 1017.92 | 1016.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:30:00 | 1015.70 | 1017.92 | 1016.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 1013.90 | 1017.12 | 1016.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 1011.65 | 1017.12 | 1016.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 1011.35 | 1015.07 | 1015.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 13:15:00 | 1008.25 | 1012.14 | 1013.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 09:15:00 | 1015.50 | 1010.73 | 1010.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1015.50 | 1010.73 | 1010.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1015.50 | 1010.73 | 1010.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 1015.80 | 1010.73 | 1010.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 1013.15 | 1011.22 | 1011.18 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 1010.00 | 1011.02 | 1011.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 1006.45 | 1010.10 | 1010.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 1018.50 | 1008.55 | 1009.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 1018.50 | 1008.55 | 1009.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1018.50 | 1008.55 | 1009.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 1018.50 | 1008.55 | 1009.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 1021.00 | 1011.04 | 1010.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 1034.80 | 1020.17 | 1016.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 10:15:00 | 1024.00 | 1026.55 | 1022.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 11:00:00 | 1024.00 | 1026.55 | 1022.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1025.05 | 1026.99 | 1023.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:30:00 | 1026.25 | 1026.99 | 1023.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1023.90 | 1026.38 | 1023.94 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1015.60 | 1023.19 | 1023.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 1011.05 | 1018.56 | 1020.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 09:15:00 | 999.75 | 995.73 | 1000.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 999.75 | 995.73 | 1000.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 999.75 | 995.73 | 1000.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 1002.35 | 995.73 | 1000.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 996.25 | 995.83 | 999.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 986.30 | 995.63 | 999.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 993.30 | 995.16 | 998.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 993.80 | 995.43 | 998.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 1005.20 | 997.84 | 999.04 | SL hit (close>static) qty=1.00 sl=1000.15 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1004.65 | 1000.67 | 1000.17 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 15:15:00 | 997.00 | 999.75 | 999.88 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 1001.95 | 1000.15 | 999.94 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 999.45 | 999.77 | 999.81 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1005.00 | 1000.64 | 1000.19 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 999.50 | 1000.41 | 1000.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 997.40 | 999.34 | 999.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 989.20 | 982.20 | 986.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 989.20 | 982.20 | 986.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 989.20 | 982.20 | 986.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 989.20 | 982.20 | 986.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 985.00 | 982.76 | 986.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 979.00 | 984.26 | 986.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 978.60 | 984.15 | 986.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 992.00 | 986.06 | 986.54 | SL hit (close>static) qty=1.00 sl=990.80 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 993.50 | 987.55 | 987.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 998.45 | 990.48 | 988.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 11:15:00 | 997.40 | 997.70 | 994.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:00:00 | 997.40 | 997.70 | 994.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 989.75 | 996.11 | 994.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:00:00 | 989.75 | 996.11 | 994.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 990.70 | 995.03 | 993.99 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 989.90 | 993.29 | 993.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 983.75 | 991.38 | 992.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 992.55 | 991.61 | 992.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 10:15:00 | 992.55 | 991.61 | 992.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 992.55 | 991.61 | 992.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 992.55 | 991.61 | 992.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 990.65 | 991.42 | 992.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 990.00 | 991.83 | 992.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 13:00:00 | 989.50 | 990.91 | 991.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 994.00 | 991.47 | 991.80 | SL hit (close>static) qty=1.00 sl=993.75 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 15:15:00 | 996.00 | 992.37 | 992.19 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 987.50 | 991.40 | 991.76 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 1003.30 | 993.37 | 992.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 1024.95 | 1003.54 | 999.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 1006.85 | 1011.57 | 1006.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 1006.85 | 1011.57 | 1006.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1000.00 | 1009.25 | 1005.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1008.00 | 1009.25 | 1005.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-28 09:15:00 | 1108.80 | 1082.68 | 1063.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 1061.15 | 1071.53 | 1071.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 10:15:00 | 1058.05 | 1067.74 | 1069.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 12:15:00 | 1072.80 | 1067.36 | 1069.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 12:15:00 | 1072.80 | 1067.36 | 1069.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 1072.80 | 1067.36 | 1069.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:00:00 | 1072.80 | 1067.36 | 1069.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1077.00 | 1069.29 | 1069.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:30:00 | 1074.05 | 1069.29 | 1069.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1068.50 | 1067.70 | 1068.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 1056.70 | 1067.70 | 1068.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:45:00 | 1060.30 | 1057.20 | 1061.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 1067.65 | 1063.60 | 1063.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 1067.65 | 1063.60 | 1063.41 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 1057.85 | 1062.37 | 1062.94 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 1083.15 | 1063.88 | 1062.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 10:15:00 | 1097.85 | 1080.92 | 1073.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 1093.05 | 1100.17 | 1088.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 1093.05 | 1100.17 | 1088.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1093.05 | 1100.17 | 1088.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 1093.05 | 1100.17 | 1088.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1086.80 | 1097.50 | 1088.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 1086.80 | 1097.50 | 1088.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1078.35 | 1093.67 | 1087.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:30:00 | 1077.00 | 1093.67 | 1087.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1078.90 | 1086.68 | 1085.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1078.90 | 1086.68 | 1085.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 1077.15 | 1084.77 | 1084.81 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1089.50 | 1085.72 | 1085.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 1104.95 | 1089.57 | 1087.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 14:15:00 | 1091.90 | 1094.00 | 1090.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 14:15:00 | 1091.90 | 1094.00 | 1090.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 1091.90 | 1094.00 | 1090.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 1091.90 | 1094.00 | 1090.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1090.50 | 1093.30 | 1090.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 1092.10 | 1093.30 | 1090.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1091.70 | 1092.98 | 1090.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:45:00 | 1090.20 | 1092.98 | 1090.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 1098.20 | 1094.02 | 1091.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:45:00 | 1086.65 | 1094.02 | 1091.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1080.00 | 1091.84 | 1090.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1080.00 | 1091.84 | 1090.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1080.90 | 1089.65 | 1090.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 15:15:00 | 1074.05 | 1086.53 | 1088.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 1093.10 | 1087.57 | 1088.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 10:15:00 | 1093.10 | 1087.57 | 1088.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1093.10 | 1087.57 | 1088.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 1093.10 | 1087.57 | 1088.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1084.25 | 1086.91 | 1088.29 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 1094.00 | 1088.52 | 1088.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 1097.10 | 1090.95 | 1089.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 1089.00 | 1090.92 | 1089.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 1089.00 | 1090.92 | 1089.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1089.00 | 1090.92 | 1089.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:30:00 | 1092.60 | 1090.92 | 1089.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1090.05 | 1090.75 | 1089.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1107.80 | 1090.75 | 1089.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 12:15:00 | 1088.10 | 1092.21 | 1092.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 1088.10 | 1092.21 | 1092.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 13:15:00 | 1084.60 | 1090.69 | 1091.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1079.40 | 1074.58 | 1080.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1079.40 | 1074.58 | 1080.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1079.40 | 1074.58 | 1080.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 1095.50 | 1074.58 | 1080.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1064.35 | 1072.54 | 1079.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 1060.50 | 1072.54 | 1079.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 1061.00 | 1067.57 | 1075.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 1082.70 | 1067.85 | 1072.74 | SL hit (close>static) qty=1.00 sl=1079.95 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 1087.30 | 1076.45 | 1075.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1114.30 | 1085.75 | 1080.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 14:15:00 | 1106.10 | 1108.57 | 1097.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 15:00:00 | 1106.10 | 1108.57 | 1097.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1103.25 | 1106.73 | 1099.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:45:00 | 1101.65 | 1106.73 | 1099.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 1102.10 | 1104.82 | 1100.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:30:00 | 1103.00 | 1104.82 | 1100.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1098.40 | 1103.54 | 1100.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 1098.40 | 1103.54 | 1100.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1098.10 | 1102.45 | 1099.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 1108.90 | 1102.45 | 1099.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1091.00 | 1100.16 | 1099.17 | SL hit (close<static) qty=1.00 sl=1096.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 1089.50 | 1098.03 | 1098.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 1086.30 | 1095.68 | 1097.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 1094.00 | 1092.16 | 1094.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 1094.00 | 1092.16 | 1094.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1094.00 | 1092.16 | 1094.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 1094.00 | 1092.16 | 1094.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1095.90 | 1092.91 | 1094.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 1095.90 | 1092.91 | 1094.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1099.55 | 1094.23 | 1095.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:15:00 | 1094.20 | 1094.56 | 1095.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 1094.00 | 1094.36 | 1094.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1123.00 | 1100.50 | 1097.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1123.00 | 1100.50 | 1097.66 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 1095.40 | 1104.53 | 1105.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 12:15:00 | 1092.70 | 1100.86 | 1103.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1076.90 | 1073.00 | 1081.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:45:00 | 1075.90 | 1073.00 | 1081.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1069.75 | 1072.35 | 1080.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:45:00 | 1064.70 | 1070.65 | 1079.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 1055.00 | 1050.41 | 1050.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1055.00 | 1050.41 | 1050.32 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 1045.95 | 1049.52 | 1049.92 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 1079.65 | 1055.78 | 1052.71 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 1050.60 | 1055.81 | 1056.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 1049.80 | 1054.61 | 1055.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 12:15:00 | 1058.95 | 1054.64 | 1055.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 12:15:00 | 1058.95 | 1054.64 | 1055.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 1058.95 | 1054.64 | 1055.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:00:00 | 1058.95 | 1054.64 | 1055.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1055.00 | 1054.71 | 1055.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:45:00 | 1058.30 | 1054.71 | 1055.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1050.75 | 1053.92 | 1055.13 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 1058.45 | 1056.10 | 1055.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 13:15:00 | 1069.75 | 1060.28 | 1058.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1064.15 | 1065.78 | 1061.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1064.15 | 1065.78 | 1061.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1064.15 | 1065.78 | 1061.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1064.15 | 1065.78 | 1061.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1074.85 | 1067.60 | 1062.81 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1056.25 | 1061.29 | 1061.87 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1071.60 | 1062.62 | 1062.08 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1058.25 | 1061.50 | 1061.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 1052.90 | 1059.32 | 1060.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 11:15:00 | 1044.35 | 1036.57 | 1043.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 11:15:00 | 1044.35 | 1036.57 | 1043.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1044.35 | 1036.57 | 1043.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1045.00 | 1036.57 | 1043.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1050.05 | 1039.27 | 1044.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 1050.75 | 1039.27 | 1044.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1054.05 | 1042.22 | 1045.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 1054.05 | 1042.22 | 1045.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1041.30 | 1040.94 | 1043.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 1032.20 | 1036.19 | 1040.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 17:15:00 | 1020.80 | 1009.29 | 1008.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 1020.80 | 1009.29 | 1008.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 1029.00 | 1013.23 | 1010.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1008.70 | 1012.33 | 1010.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1008.70 | 1012.33 | 1010.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1008.70 | 1012.33 | 1010.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1008.70 | 1012.33 | 1010.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1005.00 | 1010.86 | 1009.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 1003.70 | 1010.86 | 1009.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1008.95 | 1010.48 | 1009.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:15:00 | 1010.20 | 1009.77 | 1009.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:45:00 | 1010.65 | 1010.11 | 1009.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 1013.40 | 1010.14 | 1009.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:45:00 | 1012.00 | 1010.31 | 1009.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1006.95 | 1009.64 | 1009.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:30:00 | 1005.00 | 1009.64 | 1009.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 1007.60 | 1009.23 | 1009.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 1007.60 | 1009.23 | 1009.42 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 1011.70 | 1009.72 | 1009.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1026.10 | 1013.30 | 1011.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1022.90 | 1024.99 | 1019.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1022.90 | 1024.99 | 1019.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1021.70 | 1024.33 | 1019.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1020.75 | 1024.33 | 1019.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1020.55 | 1022.28 | 1020.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 1020.55 | 1022.28 | 1020.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1014.00 | 1020.63 | 1019.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1014.25 | 1020.63 | 1019.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1012.50 | 1019.00 | 1018.83 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1011.00 | 1017.40 | 1018.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 1007.30 | 1013.53 | 1016.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1002.40 | 1001.24 | 1006.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 12:45:00 | 997.45 | 1000.06 | 1004.52 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 947.58 | 961.76 | 972.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 955.60 | 949.68 | 959.38 | SL hit (close>ema200) qty=0.50 sl=949.68 alert=retest1 |

### Cycle 49 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 954.00 | 947.21 | 946.54 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 15:15:00 | 942.95 | 946.15 | 946.29 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 09:15:00 | 948.25 | 946.57 | 946.47 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 944.50 | 946.39 | 946.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 943.65 | 945.84 | 946.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 945.50 | 944.74 | 945.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 945.50 | 944.74 | 945.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 945.50 | 944.74 | 945.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 15:00:00 | 939.95 | 943.89 | 944.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:45:00 | 940.05 | 942.88 | 944.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:15:00 | 940.90 | 942.88 | 944.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 939.40 | 937.28 | 940.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 937.00 | 937.59 | 939.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 939.35 | 937.59 | 939.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 941.10 | 938.32 | 939.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 944.85 | 938.32 | 939.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 947.95 | 940.24 | 940.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 951.70 | 940.24 | 940.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-02 10:15:00 | 949.15 | 942.03 | 941.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 949.15 | 942.03 | 941.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 969.00 | 951.19 | 946.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 14:15:00 | 957.00 | 957.35 | 951.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 957.00 | 957.35 | 951.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 953.20 | 957.87 | 953.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 953.20 | 957.87 | 953.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 952.05 | 956.71 | 953.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 953.20 | 956.71 | 953.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 952.20 | 955.81 | 953.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:30:00 | 952.70 | 955.81 | 953.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 951.10 | 954.87 | 953.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 949.80 | 954.87 | 953.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 953.50 | 954.61 | 953.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 952.40 | 954.61 | 953.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 955.30 | 954.75 | 953.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 958.05 | 954.61 | 954.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 13:15:00 | 951.60 | 953.51 | 953.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 13:15:00 | 951.60 | 953.51 | 953.66 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 955.00 | 953.81 | 953.78 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 950.30 | 953.38 | 953.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 948.45 | 951.75 | 952.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 944.00 | 943.99 | 947.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 13:00:00 | 941.45 | 943.49 | 946.08 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 950.80 | 944.61 | 945.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 950.80 | 944.61 | 945.73 | SL hit (close>ema400) qty=1.00 sl=945.73 alert=retest1 |

### Cycle 57 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 909.00 | 904.27 | 904.20 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 899.50 | 904.06 | 904.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 894.50 | 900.27 | 902.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 894.70 | 893.90 | 896.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 894.70 | 893.90 | 896.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 894.70 | 893.90 | 896.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 894.70 | 893.90 | 896.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 889.50 | 892.54 | 895.01 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 901.30 | 895.35 | 894.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 904.70 | 897.22 | 895.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 899.40 | 902.77 | 900.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 12:15:00 | 899.40 | 902.77 | 900.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 899.40 | 902.77 | 900.35 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 896.30 | 899.06 | 899.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 880.75 | 895.40 | 897.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 15:15:00 | 883.10 | 882.04 | 886.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 09:15:00 | 910.90 | 882.04 | 886.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 905.95 | 886.82 | 887.97 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 897.90 | 889.04 | 888.87 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 15:15:00 | 885.00 | 888.56 | 888.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 881.95 | 885.60 | 887.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 809.90 | 809.35 | 829.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:00:00 | 809.90 | 809.35 | 829.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 814.15 | 804.53 | 810.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 814.15 | 804.53 | 810.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 814.45 | 806.52 | 810.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:45:00 | 815.95 | 806.52 | 810.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 809.25 | 807.85 | 810.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:30:00 | 810.75 | 807.85 | 810.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 812.90 | 808.91 | 810.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:45:00 | 814.40 | 808.91 | 810.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 812.05 | 809.54 | 810.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 812.35 | 809.54 | 810.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 813.80 | 810.39 | 810.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:00:00 | 813.80 | 810.39 | 810.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 810.00 | 810.31 | 810.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:15:00 | 809.60 | 810.31 | 810.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 811.35 | 810.49 | 810.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 811.35 | 810.49 | 810.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 14:15:00 | 812.65 | 810.92 | 810.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 818.40 | 819.20 | 815.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 14:45:00 | 819.70 | 819.20 | 815.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 816.00 | 818.56 | 815.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 794.00 | 818.56 | 815.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 801.70 | 815.19 | 814.69 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 800.95 | 812.34 | 813.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 15:15:00 | 797.25 | 802.61 | 805.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 15:15:00 | 756.20 | 755.62 | 766.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:15:00 | 769.25 | 755.62 | 766.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 772.40 | 758.97 | 766.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 773.55 | 758.97 | 766.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 772.20 | 761.62 | 767.10 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 784.40 | 771.46 | 770.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 789.00 | 776.42 | 772.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 776.90 | 779.66 | 775.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 776.90 | 779.66 | 775.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 777.35 | 779.20 | 776.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 778.70 | 779.20 | 776.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 778.95 | 779.15 | 776.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 779.95 | 779.15 | 776.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:30:00 | 780.75 | 779.32 | 776.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 774.30 | 782.20 | 780.83 | SL hit (close<static) qty=1.00 sl=776.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 775.75 | 779.38 | 779.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 763.05 | 776.12 | 778.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 772.00 | 769.09 | 772.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 15:15:00 | 772.00 | 769.09 | 772.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 772.00 | 769.09 | 772.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 776.75 | 769.09 | 772.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 769.90 | 769.25 | 772.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 768.75 | 769.25 | 772.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:30:00 | 768.60 | 769.26 | 771.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 786.70 | 773.23 | 772.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 786.70 | 773.23 | 772.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 793.15 | 781.52 | 777.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 787.80 | 790.85 | 785.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 787.80 | 790.85 | 785.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 785.75 | 789.83 | 785.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 785.75 | 789.83 | 785.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 786.00 | 789.07 | 785.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 782.50 | 789.07 | 785.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 786.00 | 788.45 | 785.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 780.45 | 788.45 | 785.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 779.25 | 786.61 | 785.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:45:00 | 778.95 | 786.61 | 785.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 777.50 | 784.79 | 784.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 777.50 | 784.79 | 784.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 776.35 | 783.10 | 783.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 773.65 | 778.64 | 781.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 743.45 | 743.32 | 751.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 744.15 | 743.32 | 751.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 733.00 | 730.99 | 738.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:30:00 | 740.05 | 730.99 | 738.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 722.15 | 722.61 | 728.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 717.70 | 720.88 | 726.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 14:00:00 | 716.25 | 718.79 | 724.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 731.65 | 724.18 | 725.27 | SL hit (close>static) qty=1.00 sl=731.50 alert=retest2 |

### Cycle 69 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 730.90 | 725.80 | 725.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 742.00 | 729.70 | 727.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 743.90 | 743.95 | 739.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:45:00 | 743.95 | 743.95 | 739.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 739.15 | 742.46 | 739.55 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 733.75 | 737.43 | 737.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 730.80 | 735.12 | 736.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 665.80 | 661.55 | 679.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:45:00 | 673.25 | 661.55 | 679.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 658.85 | 654.26 | 664.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 663.90 | 654.26 | 664.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 660.75 | 657.21 | 663.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:30:00 | 659.15 | 658.10 | 663.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 675.70 | 661.62 | 664.63 | SL hit (close>static) qty=1.00 sl=664.40 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 679.25 | 667.05 | 666.69 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 668.05 | 670.04 | 670.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 662.55 | 668.54 | 669.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 643.65 | 643.53 | 650.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 643.65 | 643.53 | 650.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 646.40 | 635.95 | 639.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 644.70 | 635.95 | 639.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 646.30 | 638.02 | 639.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 646.30 | 638.02 | 639.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 646.45 | 641.37 | 641.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 656.20 | 646.86 | 644.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 665.70 | 665.83 | 659.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 665.70 | 665.83 | 659.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 697.00 | 700.21 | 694.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 696.85 | 700.21 | 694.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 692.15 | 698.60 | 694.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 692.15 | 698.60 | 694.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 694.15 | 697.71 | 694.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 694.15 | 697.71 | 694.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 699.20 | 698.21 | 695.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 696.40 | 698.21 | 695.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 684.25 | 696.01 | 695.06 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 690.35 | 693.93 | 694.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 15:15:00 | 683.00 | 689.62 | 691.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 691.25 | 689.77 | 691.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 691.25 | 689.77 | 691.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 691.25 | 689.77 | 691.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 691.25 | 689.77 | 691.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 692.35 | 690.28 | 691.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:30:00 | 691.35 | 690.28 | 691.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 689.00 | 690.03 | 691.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 684.00 | 689.70 | 691.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 14:45:00 | 684.25 | 688.05 | 690.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 15:15:00 | 687.50 | 685.19 | 685.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:00:00 | 686.90 | 685.69 | 685.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 682.85 | 685.12 | 685.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:45:00 | 681.10 | 684.09 | 685.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 13:30:00 | 680.95 | 683.21 | 684.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 672.90 | 682.28 | 683.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 653.12 | 676.71 | 681.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:15:00 | 652.55 | 671.93 | 678.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:15:00 | 649.80 | 667.78 | 676.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:15:00 | 650.04 | 667.78 | 676.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 12:15:00 | 647.04 | 663.52 | 673.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 12:15:00 | 646.90 | 663.52 | 673.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 615.60 | 645.18 | 661.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 633.50 | 622.79 | 621.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 649.60 | 629.77 | 625.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 651.75 | 652.28 | 647.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 651.75 | 652.28 | 647.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 700.80 | 715.21 | 709.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 700.80 | 715.21 | 709.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 695.90 | 711.35 | 708.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 695.90 | 711.35 | 708.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 697.45 | 704.67 | 705.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 692.00 | 700.71 | 703.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 702.55 | 701.08 | 703.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 702.55 | 701.08 | 703.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 705.20 | 701.91 | 703.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 705.20 | 701.91 | 703.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 708.10 | 703.14 | 703.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 707.45 | 703.14 | 703.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 709.45 | 704.93 | 704.68 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 667.25 | 697.62 | 701.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 657.90 | 670.34 | 683.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 662.65 | 654.87 | 662.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 662.65 | 654.87 | 662.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 662.65 | 654.87 | 662.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 660.40 | 654.87 | 662.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 664.50 | 656.80 | 662.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 667.00 | 656.80 | 662.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 663.25 | 658.09 | 662.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 663.25 | 658.09 | 662.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 663.60 | 659.19 | 662.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 664.25 | 659.19 | 662.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 664.10 | 660.17 | 663.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 664.10 | 660.17 | 663.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 665.10 | 661.16 | 663.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 667.45 | 661.16 | 663.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 666.00 | 662.13 | 663.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 662.20 | 662.13 | 663.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 656.45 | 653.52 | 656.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 658.10 | 653.52 | 656.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 656.40 | 654.10 | 656.84 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 668.65 | 658.53 | 658.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 672.75 | 663.05 | 660.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 659.95 | 662.90 | 660.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 659.95 | 662.90 | 660.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 659.95 | 662.90 | 660.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 658.95 | 662.90 | 660.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 657.00 | 661.72 | 660.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 657.00 | 661.72 | 660.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 651.80 | 659.74 | 659.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 649.50 | 659.74 | 659.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 652.00 | 658.19 | 658.99 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 13:15:00 | 661.35 | 659.15 | 659.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 14:15:00 | 666.25 | 660.57 | 659.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 753.70 | 757.98 | 748.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 12:00:00 | 753.70 | 757.98 | 748.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 746.70 | 754.69 | 748.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 746.70 | 754.69 | 748.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 745.25 | 752.80 | 747.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 743.45 | 752.80 | 747.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 744.00 | 751.04 | 747.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 742.30 | 751.04 | 747.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 757.90 | 753.48 | 749.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 760.50 | 757.31 | 753.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 757.00 | 766.24 | 766.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 757.00 | 766.24 | 766.88 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 770.15 | 766.97 | 766.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 773.05 | 768.18 | 767.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 772.10 | 775.41 | 772.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 772.10 | 775.41 | 772.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 772.10 | 775.41 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 772.10 | 775.41 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 771.70 | 774.67 | 771.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 776.80 | 774.67 | 771.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 776.25 | 774.98 | 772.37 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 767.20 | 771.46 | 772.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 757.75 | 767.78 | 770.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 763.35 | 762.41 | 765.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 763.35 | 762.41 | 765.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 763.35 | 762.41 | 765.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 759.15 | 764.08 | 765.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 771.75 | 761.75 | 762.39 | SL hit (close>static) qty=1.00 sl=765.50 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 772.20 | 763.84 | 763.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 774.85 | 767.57 | 765.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 783.00 | 787.70 | 782.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 783.00 | 787.70 | 782.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 783.00 | 787.70 | 782.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 783.40 | 787.70 | 782.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 785.95 | 787.35 | 782.60 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 775.05 | 779.96 | 780.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 764.15 | 776.80 | 778.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 751.95 | 748.44 | 756.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 751.95 | 748.44 | 756.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 732.75 | 734.56 | 740.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 727.70 | 733.48 | 739.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 721.85 | 706.30 | 704.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 721.85 | 706.30 | 704.62 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 703.00 | 708.83 | 709.41 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 710.55 | 708.81 | 708.65 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 707.35 | 708.52 | 708.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 703.25 | 706.10 | 707.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 704.85 | 702.33 | 703.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 704.85 | 702.33 | 703.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 704.85 | 702.33 | 703.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 704.85 | 702.33 | 703.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 706.90 | 703.24 | 704.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 707.00 | 703.24 | 704.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 704.50 | 704.12 | 704.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 703.00 | 704.12 | 704.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 711.30 | 705.50 | 704.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 711.30 | 705.50 | 704.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 714.80 | 710.17 | 708.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 709.95 | 711.67 | 709.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 709.95 | 711.67 | 709.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 708.20 | 710.98 | 709.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 705.00 | 710.98 | 709.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 709.00 | 710.58 | 709.25 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 706.00 | 708.20 | 708.37 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 710.35 | 708.63 | 708.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 715.80 | 710.07 | 709.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 730.10 | 732.91 | 727.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 730.45 | 732.91 | 727.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 727.60 | 731.51 | 727.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 727.60 | 731.51 | 727.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 725.30 | 730.27 | 727.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 725.30 | 730.27 | 727.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 719.50 | 725.75 | 726.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 715.70 | 721.95 | 724.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 716.15 | 716.15 | 719.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 716.15 | 716.15 | 719.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 715.00 | 715.95 | 718.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 711.40 | 715.13 | 716.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 711.35 | 714.65 | 716.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 718.70 | 716.02 | 715.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 718.70 | 716.02 | 715.87 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 709.80 | 714.85 | 715.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 708.65 | 713.31 | 714.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 712.70 | 711.73 | 713.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 712.70 | 711.73 | 713.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 712.70 | 711.73 | 713.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 712.75 | 711.73 | 713.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 709.90 | 711.37 | 713.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 707.05 | 710.10 | 712.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 707.60 | 704.78 | 706.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:00:00 | 707.10 | 705.64 | 706.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 711.00 | 707.69 | 707.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 711.00 | 707.69 | 707.56 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 703.25 | 707.61 | 707.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 695.20 | 703.54 | 705.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 694.70 | 693.57 | 697.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:30:00 | 693.60 | 693.57 | 697.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 690.05 | 692.90 | 696.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 689.00 | 691.60 | 694.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 687.40 | 690.42 | 693.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 654.55 | 664.95 | 674.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 653.03 | 664.95 | 674.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 667.20 | 665.40 | 673.60 | SL hit (close>ema200) qty=0.50 sl=665.40 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 669.55 | 664.15 | 663.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 674.65 | 666.95 | 665.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 667.75 | 668.14 | 666.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 667.75 | 668.14 | 666.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 670.05 | 668.52 | 666.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 668.00 | 668.52 | 666.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 685.10 | 687.71 | 685.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 684.40 | 687.71 | 685.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 683.90 | 686.95 | 685.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:15:00 | 684.90 | 686.95 | 685.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 685.00 | 686.56 | 685.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 687.25 | 686.43 | 685.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 688.15 | 685.48 | 685.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:30:00 | 687.00 | 687.62 | 686.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:45:00 | 687.20 | 687.24 | 686.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 685.00 | 686.79 | 686.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 680.00 | 686.79 | 686.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 681.50 | 685.74 | 685.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 681.50 | 685.74 | 685.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 671.85 | 679.18 | 682.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 665.50 | 663.28 | 668.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 665.50 | 663.28 | 668.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 665.50 | 663.28 | 668.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 668.35 | 663.28 | 668.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 671.65 | 665.39 | 668.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 671.65 | 665.39 | 668.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 672.70 | 666.85 | 668.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 672.70 | 666.85 | 668.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 676.60 | 670.06 | 670.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 678.40 | 671.73 | 670.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 681.35 | 681.81 | 678.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 681.35 | 681.81 | 678.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 680.55 | 681.50 | 678.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 678.90 | 681.50 | 678.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 682.50 | 682.80 | 681.03 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 670.75 | 678.42 | 679.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 669.65 | 676.67 | 678.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 674.00 | 673.44 | 675.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 15:15:00 | 674.00 | 673.44 | 675.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 674.00 | 673.44 | 675.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 676.75 | 673.44 | 675.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 680.95 | 674.94 | 676.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 679.80 | 674.94 | 676.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 681.00 | 676.15 | 676.75 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 681.60 | 677.24 | 677.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 686.75 | 679.89 | 678.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 680.25 | 680.46 | 679.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 680.25 | 680.46 | 679.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 680.40 | 680.44 | 679.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 680.30 | 680.44 | 679.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 679.60 | 680.28 | 679.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 679.50 | 680.28 | 679.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 680.95 | 680.41 | 679.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 688.40 | 680.41 | 679.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 705.50 | 713.77 | 714.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 705.50 | 713.77 | 714.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 702.85 | 709.00 | 712.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 677.30 | 672.68 | 677.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 677.30 | 672.68 | 677.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 677.30 | 672.68 | 677.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 678.85 | 672.68 | 677.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 676.50 | 673.45 | 677.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 673.55 | 675.16 | 676.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 679.00 | 673.51 | 674.87 | SL hit (close>static) qty=1.00 sl=677.60 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 684.45 | 677.04 | 676.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 691.30 | 679.89 | 677.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 711.95 | 712.72 | 707.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 11:15:00 | 708.00 | 711.26 | 707.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 708.00 | 711.26 | 707.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 708.40 | 711.26 | 707.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 709.40 | 710.89 | 707.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 711.45 | 710.89 | 707.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 712.05 | 709.61 | 707.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 706.15 | 713.08 | 712.53 | SL hit (close<static) qty=1.00 sl=706.50 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 701.50 | 710.76 | 711.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 695.35 | 702.67 | 706.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 695.70 | 693.69 | 699.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 695.70 | 693.69 | 699.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 697.65 | 695.09 | 697.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 697.25 | 695.09 | 697.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 691.50 | 694.37 | 697.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 689.20 | 693.14 | 695.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:45:00 | 690.75 | 692.82 | 695.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:45:00 | 689.55 | 692.22 | 694.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 698.70 | 688.31 | 687.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 698.70 | 688.31 | 687.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 700.00 | 695.66 | 693.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 695.00 | 695.81 | 694.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 12:00:00 | 695.00 | 695.81 | 694.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 694.80 | 695.53 | 694.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 694.80 | 695.53 | 694.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 694.85 | 695.39 | 694.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 693.80 | 695.39 | 694.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 699.45 | 699.54 | 698.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 698.30 | 699.54 | 698.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 697.50 | 698.90 | 698.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 697.50 | 698.90 | 698.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 697.20 | 698.56 | 698.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:15:00 | 697.35 | 698.56 | 698.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 696.20 | 698.09 | 698.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 696.20 | 698.09 | 698.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 695.50 | 697.57 | 697.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 691.30 | 696.32 | 697.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 696.40 | 692.88 | 694.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 696.40 | 692.88 | 694.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 696.40 | 692.88 | 694.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 696.40 | 692.88 | 694.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 697.40 | 693.78 | 694.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 695.00 | 693.78 | 694.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 677.50 | 674.89 | 678.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 677.35 | 674.89 | 678.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 679.70 | 675.85 | 678.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 681.35 | 675.85 | 678.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 676.90 | 676.06 | 678.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 676.25 | 676.10 | 678.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 676.35 | 677.42 | 678.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 676.30 | 677.20 | 678.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 682.00 | 678.52 | 678.57 | SL hit (close>static) qty=1.00 sl=680.65 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 684.95 | 679.81 | 679.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 696.65 | 683.83 | 681.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 693.25 | 694.93 | 690.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 692.80 | 694.93 | 690.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 690.35 | 694.01 | 690.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 690.35 | 694.01 | 690.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 686.20 | 692.45 | 690.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 686.20 | 692.45 | 690.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 686.00 | 691.16 | 689.97 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 681.05 | 689.14 | 689.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 678.35 | 685.14 | 687.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 12:15:00 | 683.20 | 682.32 | 684.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 12:45:00 | 683.55 | 682.32 | 684.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 677.35 | 677.74 | 680.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 680.30 | 677.74 | 680.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 678.15 | 677.82 | 680.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 679.50 | 677.82 | 680.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 683.90 | 679.04 | 680.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 683.90 | 679.04 | 680.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 683.00 | 679.83 | 680.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 682.30 | 679.83 | 680.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:15:00 | 682.40 | 680.43 | 680.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 15:15:00 | 682.75 | 681.24 | 681.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 682.75 | 681.24 | 681.13 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 673.85 | 679.90 | 680.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 670.75 | 675.11 | 677.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 676.75 | 674.66 | 677.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 676.75 | 674.66 | 677.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 676.75 | 674.66 | 677.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 678.30 | 674.66 | 677.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 678.90 | 675.51 | 677.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 678.90 | 675.51 | 677.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 679.00 | 676.21 | 677.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 681.35 | 676.21 | 677.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 677.00 | 677.29 | 677.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 677.00 | 677.29 | 677.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 676.85 | 677.20 | 677.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 673.75 | 677.20 | 677.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 675.15 | 673.17 | 674.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 679.95 | 675.81 | 675.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 679.95 | 675.81 | 675.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 681.60 | 677.75 | 676.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 679.35 | 679.95 | 678.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 679.35 | 679.95 | 678.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 677.00 | 679.36 | 677.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 677.00 | 679.36 | 677.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 674.60 | 678.41 | 677.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 674.60 | 678.41 | 677.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 677.50 | 677.68 | 677.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 679.15 | 677.68 | 677.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 679.00 | 678.23 | 677.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 680.00 | 679.19 | 678.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 11:15:00 | 675.05 | 678.27 | 678.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 675.05 | 678.27 | 678.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 672.50 | 675.87 | 676.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 677.65 | 674.91 | 676.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 677.65 | 674.91 | 676.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 677.65 | 674.91 | 676.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 677.45 | 674.91 | 676.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 675.55 | 675.04 | 676.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:45:00 | 674.40 | 675.08 | 676.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:30:00 | 675.10 | 675.63 | 676.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 674.45 | 675.40 | 676.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 674.70 | 675.42 | 675.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 666.50 | 673.52 | 675.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 664.90 | 669.32 | 672.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 663.00 | 667.24 | 670.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 641.35 | 654.72 | 661.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 15:15:00 | 640.68 | 646.74 | 651.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 15:15:00 | 640.73 | 646.74 | 651.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 15:15:00 | 640.97 | 646.74 | 651.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 647.50 | 646.89 | 651.46 | SL hit (close>ema200) qty=0.50 sl=646.89 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 655.60 | 652.47 | 652.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 659.70 | 654.89 | 653.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 656.60 | 657.20 | 655.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 656.60 | 657.20 | 655.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 657.50 | 657.26 | 655.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 656.55 | 657.26 | 655.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 651.05 | 656.02 | 655.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 651.05 | 656.02 | 655.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 651.20 | 655.05 | 655.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 650.00 | 653.35 | 654.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 644.75 | 643.95 | 647.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 645.35 | 643.95 | 647.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 644.80 | 643.41 | 645.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 646.85 | 643.41 | 645.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 646.20 | 643.97 | 645.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 648.10 | 643.97 | 645.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 647.85 | 644.74 | 645.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 647.25 | 644.74 | 645.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 646.90 | 645.18 | 646.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 648.00 | 645.18 | 646.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 652.65 | 646.67 | 646.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 653.80 | 648.10 | 647.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 661.30 | 662.70 | 659.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 664.15 | 662.70 | 659.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 659.75 | 661.68 | 660.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 658.80 | 661.68 | 660.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 658.00 | 660.94 | 659.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 657.40 | 660.94 | 659.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 659.15 | 660.01 | 659.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 659.25 | 660.01 | 659.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 658.80 | 659.77 | 659.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 658.85 | 659.77 | 659.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 658.40 | 659.49 | 659.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 657.20 | 659.04 | 659.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 643.00 | 642.69 | 646.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 643.00 | 642.69 | 646.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 641.80 | 642.74 | 645.28 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 652.45 | 645.21 | 645.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 653.20 | 646.81 | 645.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 651.85 | 653.61 | 651.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 651.85 | 653.61 | 651.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 651.25 | 653.14 | 651.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 649.85 | 653.14 | 651.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 650.50 | 652.61 | 650.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 649.55 | 652.61 | 650.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 650.50 | 652.19 | 650.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:15:00 | 649.05 | 652.19 | 650.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 648.40 | 651.43 | 650.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:15:00 | 647.85 | 651.43 | 650.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 647.55 | 650.12 | 650.23 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 682.00 | 656.24 | 652.96 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 655.70 | 663.06 | 663.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 654.80 | 661.41 | 663.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 652.25 | 651.45 | 655.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 654.30 | 651.45 | 655.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 650.10 | 651.25 | 654.66 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 659.85 | 655.44 | 655.39 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 651.30 | 654.61 | 655.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 11:15:00 | 648.25 | 652.84 | 654.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 655.45 | 650.76 | 652.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 655.45 | 650.76 | 652.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 655.45 | 650.76 | 652.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 656.50 | 650.76 | 652.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 653.30 | 651.27 | 652.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 652.80 | 651.27 | 652.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 654.00 | 651.82 | 652.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:30:00 | 652.65 | 651.82 | 652.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 650.70 | 651.59 | 652.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 655.00 | 651.59 | 652.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 649.90 | 651.25 | 652.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 651.35 | 651.25 | 652.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 650.35 | 651.07 | 651.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 651.90 | 651.07 | 651.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 651.45 | 651.15 | 651.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 669.10 | 651.15 | 651.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 676.35 | 656.19 | 654.15 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 650.80 | 657.13 | 657.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 647.10 | 655.13 | 656.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 659.00 | 647.06 | 649.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 659.00 | 647.06 | 649.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 659.00 | 647.06 | 649.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 660.40 | 647.06 | 649.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 658.80 | 649.41 | 650.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 658.80 | 649.41 | 650.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 656.40 | 650.80 | 650.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 659.75 | 653.51 | 652.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 654.10 | 654.25 | 652.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 654.10 | 654.25 | 652.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 659.90 | 655.38 | 653.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 650.10 | 655.38 | 653.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 655.10 | 655.72 | 654.17 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 648.05 | 652.66 | 653.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 645.30 | 651.19 | 652.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 651.45 | 651.24 | 652.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 651.45 | 651.24 | 652.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 652.70 | 651.53 | 652.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 654.90 | 651.53 | 652.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 651.30 | 651.49 | 652.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 650.10 | 651.49 | 652.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 657.60 | 653.10 | 652.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 657.60 | 653.10 | 652.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 661.40 | 655.37 | 653.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 650.15 | 655.07 | 654.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 650.15 | 655.07 | 654.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 650.15 | 655.07 | 654.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 650.15 | 655.07 | 654.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 648.25 | 653.71 | 653.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 648.65 | 653.71 | 653.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 647.70 | 652.51 | 653.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 646.50 | 649.69 | 651.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 647.55 | 647.40 | 649.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:45:00 | 646.75 | 647.40 | 649.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 651.05 | 648.13 | 649.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 651.05 | 648.13 | 649.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 647.55 | 648.01 | 649.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 647.55 | 648.01 | 649.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 646.40 | 641.24 | 643.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 646.40 | 641.24 | 643.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 646.85 | 642.37 | 643.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 657.50 | 642.37 | 643.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 657.60 | 647.39 | 646.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 660.55 | 650.02 | 647.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 649.70 | 654.97 | 651.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 649.70 | 654.97 | 651.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 649.70 | 654.97 | 651.42 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 646.35 | 649.57 | 649.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 643.10 | 648.27 | 649.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 629.50 | 622.38 | 626.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 629.50 | 622.38 | 626.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 629.50 | 622.38 | 626.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:45:00 | 630.75 | 622.38 | 626.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 633.15 | 624.53 | 627.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 634.00 | 624.53 | 627.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 632.10 | 629.02 | 628.99 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 625.00 | 628.58 | 628.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 618.30 | 625.51 | 627.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 597.95 | 597.53 | 603.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 601.10 | 597.53 | 603.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 608.25 | 599.68 | 603.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 608.30 | 599.68 | 603.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 605.90 | 600.92 | 603.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:15:00 | 604.25 | 604.81 | 605.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 603.10 | 604.78 | 605.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 603.00 | 603.87 | 604.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 603.35 | 603.60 | 604.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 603.35 | 603.55 | 604.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 607.80 | 603.55 | 604.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 608.85 | 604.61 | 604.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 608.85 | 604.61 | 604.46 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 599.50 | 604.31 | 604.62 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 607.30 | 604.91 | 604.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 608.35 | 606.07 | 605.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 606.00 | 606.66 | 605.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 15:15:00 | 606.00 | 606.66 | 605.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 606.00 | 606.66 | 605.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 609.10 | 606.66 | 605.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 602.95 | 605.89 | 605.74 | SL hit (close<static) qty=1.00 sl=605.70 alert=retest2 |

### Cycle 138 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 601.65 | 605.04 | 605.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 15:15:00 | 598.35 | 602.19 | 603.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 581.00 | 580.75 | 587.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 581.00 | 580.75 | 587.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 586.40 | 580.40 | 584.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 585.35 | 580.40 | 584.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 583.80 | 581.08 | 584.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 584.20 | 581.08 | 584.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 582.60 | 580.35 | 583.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 582.60 | 580.35 | 583.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 584.00 | 581.08 | 583.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 589.80 | 581.08 | 583.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 584.20 | 581.70 | 583.30 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 587.00 | 584.35 | 584.18 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 578.20 | 583.49 | 583.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 575.20 | 581.83 | 583.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 579.75 | 579.56 | 581.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:30:00 | 579.55 | 579.56 | 581.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 582.50 | 580.15 | 581.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 573.10 | 580.15 | 581.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 577.60 | 576.25 | 576.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 578.10 | 576.84 | 576.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 578.10 | 576.84 | 576.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 578.55 | 577.18 | 576.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 576.50 | 577.04 | 576.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 576.50 | 577.04 | 576.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 576.50 | 577.04 | 576.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 576.50 | 577.04 | 576.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 578.00 | 577.23 | 576.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 561.75 | 577.23 | 576.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 562.55 | 574.30 | 575.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 556.35 | 562.20 | 565.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 529.85 | 529.84 | 537.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:45:00 | 530.50 | 529.84 | 537.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 546.00 | 531.18 | 534.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 546.00 | 531.18 | 534.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 558.55 | 536.66 | 536.48 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 528.05 | 538.46 | 539.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 525.70 | 535.91 | 538.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 533.65 | 530.78 | 534.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 533.65 | 530.78 | 534.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 533.65 | 530.78 | 534.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 532.85 | 530.78 | 534.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 537.00 | 532.03 | 534.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 537.00 | 532.03 | 534.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 537.45 | 533.11 | 534.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 537.30 | 533.11 | 534.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 538.65 | 534.22 | 535.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 537.90 | 534.22 | 535.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 539.50 | 536.63 | 536.30 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 529.75 | 534.95 | 535.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 522.10 | 531.31 | 533.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 532.35 | 526.26 | 529.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 532.35 | 526.26 | 529.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 532.35 | 526.26 | 529.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 532.35 | 526.26 | 529.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 534.10 | 527.83 | 529.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 534.60 | 527.83 | 529.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 546.25 | 532.96 | 531.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 547.90 | 535.95 | 533.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 541.15 | 541.86 | 538.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 539.40 | 541.86 | 538.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 538.50 | 541.19 | 538.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 538.00 | 541.19 | 538.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 541.70 | 541.29 | 538.43 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 522.50 | 535.38 | 536.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 518.40 | 531.98 | 534.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 537.40 | 522.61 | 527.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 537.40 | 522.61 | 527.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 537.40 | 522.61 | 527.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 537.40 | 522.61 | 527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 533.55 | 524.80 | 528.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 530.75 | 529.72 | 529.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 531.20 | 530.14 | 530.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 531.20 | 530.14 | 530.05 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 519.00 | 527.91 | 529.05 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 540.00 | 530.52 | 529.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 541.80 | 532.78 | 530.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 555.90 | 556.92 | 551.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 555.90 | 556.92 | 551.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 556.10 | 562.47 | 559.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 563.60 | 562.40 | 560.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 572.35 | 561.25 | 560.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 575.50 | 581.27 | 581.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 575.50 | 581.27 | 581.58 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 584.70 | 581.96 | 581.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 585.80 | 583.17 | 582.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 582.90 | 583.72 | 582.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 582.90 | 583.72 | 582.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 582.90 | 583.72 | 582.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 582.90 | 583.72 | 582.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 584.85 | 583.95 | 583.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 574.40 | 583.95 | 583.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 577.80 | 582.72 | 582.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:30:00 | 578.25 | 582.72 | 582.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 575.05 | 581.19 | 581.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 573.65 | 579.68 | 581.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 562.90 | 562.30 | 567.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 13:45:00 | 563.00 | 562.30 | 567.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 571.10 | 564.06 | 568.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 571.10 | 564.06 | 568.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 565.00 | 564.25 | 567.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 576.25 | 564.25 | 567.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 575.10 | 566.42 | 568.47 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 580.15 | 570.62 | 570.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 588.55 | 574.20 | 571.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 582.00 | 583.60 | 577.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:00:00 | 582.00 | 583.60 | 577.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 579.15 | 582.49 | 578.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 579.15 | 582.49 | 578.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 577.65 | 581.52 | 578.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 577.65 | 581.52 | 578.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 573.40 | 579.90 | 577.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 573.40 | 579.90 | 577.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 569.00 | 577.72 | 577.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 569.00 | 577.72 | 577.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 570.90 | 576.35 | 576.47 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 588.80 | 576.85 | 575.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 638.05 | 596.17 | 586.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 625.00 | 625.97 | 617.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 625.00 | 625.97 | 617.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 626.75 | 629.66 | 626.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 10:45:00 | 634.05 | 630.79 | 627.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-18 09:15:00 | 1052.00 | 2024-05-23 13:15:00 | 1049.80 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-03 10:30:00 | 1038.85 | 2024-06-04 12:15:00 | 986.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1039.15 | 2024-06-04 12:15:00 | 987.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:30:00 | 1038.85 | 2024-06-05 10:15:00 | 1020.50 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1039.15 | 2024-06-05 10:15:00 | 1020.50 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2024-06-20 09:15:00 | 1010.00 | 2024-06-27 09:15:00 | 1008.50 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-07-23 12:15:00 | 986.30 | 2024-07-24 09:15:00 | 1005.20 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-07-23 13:00:00 | 993.30 | 2024-07-24 09:15:00 | 1005.20 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-07-23 13:30:00 | 993.80 | 2024-07-24 09:15:00 | 1005.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-08-06 14:00:00 | 979.00 | 2024-08-07 11:15:00 | 992.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-06 14:30:00 | 978.60 | 2024-08-07 11:15:00 | 992.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-08-13 10:00:00 | 990.00 | 2024-08-13 14:15:00 | 994.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-08-13 13:00:00 | 989.50 | 2024-08-13 14:15:00 | 994.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1008.00 | 2024-08-28 09:15:00 | 1108.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-02 10:15:00 | 1056.70 | 2024-09-03 14:15:00 | 1067.65 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-03 10:45:00 | 1060.30 | 2024-09-03 14:15:00 | 1067.65 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1107.80 | 2024-09-17 12:15:00 | 1088.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-09-19 11:15:00 | 1060.50 | 2024-09-20 09:15:00 | 1082.70 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-09-19 12:45:00 | 1061.00 | 2024-09-20 09:15:00 | 1082.70 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-09-25 09:15:00 | 1108.90 | 2024-09-25 09:15:00 | 1091.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-09-26 13:15:00 | 1094.20 | 2024-09-27 09:15:00 | 1123.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-09-26 15:00:00 | 1094.00 | 2024-09-27 09:15:00 | 1123.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-10-04 13:45:00 | 1064.70 | 2024-10-09 13:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-10-24 13:45:00 | 1032.20 | 2024-11-01 17:15:00 | 1020.80 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2024-11-04 14:15:00 | 1010.20 | 2024-11-05 12:15:00 | 1007.60 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-11-04 14:45:00 | 1010.65 | 2024-11-05 12:15:00 | 1007.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-11-05 09:45:00 | 1013.40 | 2024-11-05 12:15:00 | 1007.60 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-11-05 10:45:00 | 1012.00 | 2024-11-05 12:15:00 | 1007.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-11-12 12:45:00 | 997.45 | 2024-11-18 09:15:00 | 947.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-12 12:45:00 | 997.45 | 2024-11-19 09:15:00 | 955.60 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2024-11-21 15:00:00 | 932.90 | 2024-11-25 11:15:00 | 954.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-11-27 15:00:00 | 939.95 | 2024-12-02 10:15:00 | 949.15 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-11-28 10:45:00 | 940.05 | 2024-12-02 10:15:00 | 949.15 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-11-28 11:15:00 | 940.90 | 2024-12-02 10:15:00 | 949.15 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-11-29 09:30:00 | 939.40 | 2024-12-02 10:15:00 | 949.15 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-06 09:15:00 | 958.05 | 2024-12-06 13:15:00 | 951.60 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2024-12-11 13:00:00 | 941.45 | 2024-12-12 09:15:00 | 950.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-12-12 11:30:00 | 946.80 | 2024-12-20 13:15:00 | 899.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 11:30:00 | 946.80 | 2024-12-24 09:15:00 | 913.00 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-01-17 13:15:00 | 809.60 | 2025-01-20 13:15:00 | 811.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-01-31 09:15:00 | 779.95 | 2025-02-01 12:15:00 | 774.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-01-31 10:30:00 | 780.75 | 2025-02-01 12:15:00 | 774.30 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-02-04 10:15:00 | 768.75 | 2025-02-05 09:15:00 | 786.70 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-02-04 12:30:00 | 768.60 | 2025-02-05 09:15:00 | 786.70 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-02-18 11:45:00 | 717.70 | 2025-02-19 11:15:00 | 731.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-02-18 14:00:00 | 716.25 | 2025-02-19 11:15:00 | 731.65 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-05 13:30:00 | 659.15 | 2025-03-05 14:15:00 | 675.70 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-03-28 13:45:00 | 684.00 | 2025-04-04 09:15:00 | 653.12 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2025-03-28 14:45:00 | 684.25 | 2025-04-04 10:15:00 | 652.55 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2025-04-02 15:15:00 | 687.50 | 2025-04-04 11:15:00 | 649.80 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2025-04-03 11:00:00 | 686.90 | 2025-04-04 11:15:00 | 650.04 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2025-04-03 12:45:00 | 681.10 | 2025-04-04 12:15:00 | 647.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 13:30:00 | 680.95 | 2025-04-04 12:15:00 | 646.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:45:00 | 684.00 | 2025-04-07 09:15:00 | 615.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 14:45:00 | 684.25 | 2025-04-07 09:15:00 | 615.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-02 15:15:00 | 687.50 | 2025-04-07 09:15:00 | 618.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 11:00:00 | 686.90 | 2025-04-07 09:15:00 | 618.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 12:45:00 | 681.10 | 2025-04-07 09:15:00 | 612.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 13:30:00 | 680.95 | 2025-04-07 09:15:00 | 612.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 672.90 | 2025-04-07 09:15:00 | 605.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:30:00 | 760.50 | 2025-05-28 14:15:00 | 757.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-06 09:15:00 | 759.15 | 2025-06-09 09:15:00 | 771.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-19 11:15:00 | 727.70 | 2025-06-27 09:15:00 | 721.85 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-07-09 14:15:00 | 703.00 | 2025-07-10 09:15:00 | 711.30 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-23 09:30:00 | 711.40 | 2025-07-24 12:15:00 | 718.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-23 11:15:00 | 711.35 | 2025-07-24 12:15:00 | 718.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-28 11:30:00 | 707.05 | 2025-07-30 12:15:00 | 711.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-29 15:00:00 | 707.60 | 2025-07-30 12:15:00 | 711.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-30 10:00:00 | 707.10 | 2025-07-30 12:15:00 | 711.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-05 12:45:00 | 689.00 | 2025-08-07 13:15:00 | 654.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 15:15:00 | 687.40 | 2025-08-07 13:15:00 | 653.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 12:45:00 | 689.00 | 2025-08-07 14:15:00 | 667.20 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-08-05 15:15:00 | 687.40 | 2025-08-07 14:15:00 | 667.20 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2025-08-22 13:30:00 | 687.25 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-25 09:15:00 | 688.15 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-08-25 13:30:00 | 687.00 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-25 14:45:00 | 687.20 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-10 09:15:00 | 688.40 | 2025-09-22 11:15:00 | 705.50 | STOP_HIT | 1.00 | 2.48% |
| SELL | retest2 | 2025-09-30 13:00:00 | 673.55 | 2025-10-01 11:15:00 | 679.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-08 13:15:00 | 711.45 | 2025-10-13 09:15:00 | 706.15 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-09 09:15:00 | 712.05 | 2025-10-13 09:15:00 | 706.15 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-16 11:30:00 | 689.20 | 2025-10-23 10:15:00 | 698.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-16 12:45:00 | 690.75 | 2025-10-23 10:15:00 | 698.70 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-17 09:45:00 | 689.55 | 2025-10-23 10:15:00 | 698.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-11-10 13:00:00 | 676.25 | 2025-11-11 13:15:00 | 682.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-11 10:15:00 | 676.35 | 2025-11-11 13:15:00 | 682.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-11 11:00:00 | 676.30 | 2025-11-11 13:15:00 | 682.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-19 13:15:00 | 682.30 | 2025-11-19 15:15:00 | 682.75 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-19 14:15:00 | 682.40 | 2025-11-19 15:15:00 | 682.75 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-11-25 09:15:00 | 673.75 | 2025-11-26 12:15:00 | 679.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-26 09:30:00 | 675.15 | 2025-11-26 12:15:00 | 679.95 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-11-28 09:15:00 | 679.15 | 2025-12-02 11:15:00 | 675.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-28 13:30:00 | 679.00 | 2025-12-02 11:15:00 | 675.05 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-01 13:30:00 | 680.00 | 2025-12-02 11:15:00 | 675.05 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-04 11:45:00 | 674.40 | 2025-12-09 09:15:00 | 641.35 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-04 12:30:00 | 675.10 | 2025-12-10 15:15:00 | 640.68 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-12-04 13:30:00 | 674.45 | 2025-12-10 15:15:00 | 640.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 674.70 | 2025-12-10 15:15:00 | 640.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 11:45:00 | 674.40 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-12-04 12:30:00 | 675.10 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-04 13:30:00 | 674.45 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 674.70 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-12-05 14:45:00 | 664.90 | 2025-12-12 09:15:00 | 655.60 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2025-12-08 09:45:00 | 663.00 | 2025-12-12 09:15:00 | 655.60 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2026-01-28 10:15:00 | 650.10 | 2026-01-28 11:15:00 | 657.60 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-02-17 14:15:00 | 604.25 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-02-17 15:15:00 | 603.10 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-02-18 09:30:00 | 603.00 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-18 15:15:00 | 603.35 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-02-23 09:15:00 | 609.10 | 2026-02-23 10:15:00 | 602.95 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-03-04 09:15:00 | 573.10 | 2026-03-06 12:15:00 | 578.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-03-06 10:45:00 | 577.60 | 2026-03-06 12:15:00 | 578.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-04-01 13:30:00 | 530.75 | 2026-04-01 15:15:00 | 531.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-13 12:15:00 | 563.60 | 2026-04-20 15:15:00 | 575.50 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2026-04-15 09:15:00 | 572.35 | 2026-04-20 15:15:00 | 575.50 | STOP_HIT | 1.00 | 0.55% |
