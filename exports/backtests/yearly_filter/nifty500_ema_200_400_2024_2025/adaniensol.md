# Adani Energy Solutions Ltd. (ADANIENSOL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1351.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 10 |
| ALERT2_SKIP | 7 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 67 |
| PARTIAL | 18 |
| TARGET_HIT | 19 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 42
- **Target hits / Stop hits / Partials:** 19 / 48 / 18
- **Avg / median % per leg:** 2.20% / 0.08%
- **Sum % (uncompounded):** 186.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 7 | 17.9% | 7 | 32 | 0 | -0.23% | -8.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 39 | 7 | 17.9% | 7 | 32 | 0 | -0.23% | -8.8% |
| SELL (all) | 46 | 36 | 78.3% | 12 | 16 | 18 | 4.25% | 195.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 36 | 78.3% | 12 | 16 | 18 | 4.25% | 195.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 85 | 43 | 50.6% | 19 | 48 | 18 | 2.20% | 186.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 1122.65 | 1048.31 | 1047.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1216.40 | 1067.44 | 1058.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 998.00 | 1076.45 | 1063.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 998.00 | 1076.45 | 1063.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 998.00 | 1076.45 | 1063.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 998.00 | 1076.45 | 1063.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 977.60 | 1075.46 | 1063.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 977.60 | 1075.46 | 1063.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 1039.75 | 1056.51 | 1054.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 1039.90 | 1056.51 | 1054.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1026.10 | 1052.33 | 1052.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 11:15:00 | 1019.15 | 1050.30 | 1051.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 10:15:00 | 1043.45 | 1029.75 | 1038.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 10:15:00 | 1043.45 | 1029.75 | 1038.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1043.45 | 1029.75 | 1038.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 1059.20 | 1029.75 | 1038.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1028.10 | 1029.73 | 1038.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:45:00 | 1026.05 | 1029.72 | 1038.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 15:00:00 | 1023.95 | 1029.67 | 1038.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 09:45:00 | 1022.55 | 1029.56 | 1038.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 10:45:00 | 1025.35 | 1029.51 | 1038.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1033.35 | 1029.40 | 1038.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 1033.35 | 1029.40 | 1038.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1041.30 | 1029.59 | 1038.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 1032.00 | 1029.59 | 1038.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 09:30:00 | 1033.00 | 1020.91 | 1031.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 980.40 | 1018.95 | 1028.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 981.35 | 1018.95 | 1028.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 1023.15 | 1018.92 | 1028.52 | SL hit (close>ema200) qty=0.50 sl=1018.92 alert=retest2 |

### Cycle 3 — BUY (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 12:15:00 | 1136.70 | 1036.81 | 1036.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 09:15:00 | 1199.00 | 1041.30 | 1038.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1072.35 | 1086.48 | 1065.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1072.35 | 1086.48 | 1065.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1072.35 | 1086.48 | 1065.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:15:00 | 1081.70 | 1086.48 | 1065.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 1114.15 | 1085.18 | 1065.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1081.95 | 1085.14 | 1065.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 12:30:00 | 1080.40 | 1086.90 | 1067.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1068.40 | 1088.30 | 1071.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 1069.30 | 1088.30 | 1071.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1063.05 | 1088.05 | 1071.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 1063.05 | 1088.05 | 1071.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1065.70 | 1086.54 | 1071.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 1065.70 | 1086.54 | 1071.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 1063.15 | 1086.31 | 1071.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:45:00 | 1062.95 | 1086.31 | 1071.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 1032.50 | 1081.25 | 1069.64 | SL hit (close<static) qty=1.00 sl=1036.05 alert=retest2 |

### Cycle 4 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 1024.10 | 1059.95 | 1059.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 1018.00 | 1059.53 | 1059.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 1024.15 | 1021.89 | 1037.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 1024.15 | 1021.89 | 1037.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1024.15 | 1021.89 | 1037.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 1031.35 | 1021.89 | 1037.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1058.75 | 1022.48 | 1036.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:45:00 | 1059.05 | 1022.48 | 1036.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1045.35 | 1022.71 | 1036.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:00:00 | 1038.90 | 1024.13 | 1037.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:00:00 | 1039.85 | 1024.95 | 1037.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:30:00 | 1038.30 | 1025.50 | 1037.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:00:00 | 1039.30 | 1025.99 | 1037.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1036.20 | 1026.20 | 1037.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1036.20 | 1026.20 | 1037.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1040.00 | 1026.34 | 1037.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 1045.50 | 1026.57 | 1037.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1046.10 | 1026.77 | 1037.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 13:15:00 | 1041.00 | 1027.15 | 1037.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:45:00 | 1040.95 | 1026.11 | 1036.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:45:00 | 1039.95 | 1026.26 | 1036.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:45:00 | 1039.90 | 1026.38 | 1036.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1043.00 | 1026.82 | 1036.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1032.90 | 1026.82 | 1036.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 986.96 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 987.86 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 986.38 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 987.33 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 988.95 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 988.90 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 987.95 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 987.91 | 1026.11 | 1035.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 12:15:00 | 981.25 | 1024.87 | 1035.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 10:15:00 | 935.01 | 1021.53 | 1033.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 874.40 | 769.68 | 769.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 885.25 | 772.91 | 771.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 786.70 | 800.03 | 786.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 786.70 | 800.03 | 786.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 786.70 | 800.03 | 786.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 10:45:00 | 811.30 | 800.19 | 786.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 13:30:00 | 812.00 | 800.26 | 786.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 14:30:00 | 811.65 | 800.39 | 786.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-16 09:15:00 | 892.43 | 813.56 | 795.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 807.50 | 865.25 | 865.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 803.90 | 864.64 | 864.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 825.50 | 823.01 | 839.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:00:00 | 825.50 | 823.01 | 839.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 813.40 | 795.67 | 816.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 812.55 | 795.67 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 824.55 | 796.40 | 816.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 824.55 | 796.40 | 816.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 822.35 | 796.66 | 816.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 834.40 | 796.66 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 813.60 | 797.43 | 816.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 817.45 | 797.43 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 816.20 | 797.78 | 816.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 813.25 | 797.93 | 816.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 819.00 | 798.14 | 816.09 | SL hit (close>static) qty=1.00 sl=817.75 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 893.35 | 828.53 | 828.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 926.20 | 855.34 | 843.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 968.00 | 969.13 | 932.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:30:00 | 968.30 | 969.18 | 932.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 956.75 | 974.31 | 946.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 949.55 | 974.31 | 946.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 987.40 | 1008.26 | 982.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 984.60 | 1008.26 | 982.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 983.50 | 1008.01 | 982.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 983.50 | 1008.01 | 982.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 972.80 | 1007.66 | 982.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 972.80 | 1007.66 | 982.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 969.80 | 1007.28 | 982.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 968.00 | 1007.28 | 982.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 891.60 | 965.36 | 965.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 866.80 | 964.38 | 964.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 1029.55 | 957.75 | 957.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 1031.85 | 959.20 | 958.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 989.70 | 986.69 | 976.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 11:00:00 | 991.00 | 986.88 | 977.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 992.50 | 986.29 | 977.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 989.90 | 986.33 | 977.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 991.80 | 988.74 | 979.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 997.50 | 988.74 | 979.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 998.60 | 989.92 | 980.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:45:00 | 998.40 | 990.12 | 980.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 997.00 | 990.19 | 980.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 974.10 | 990.07 | 980.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 974.10 | 990.07 | 980.66 | SL hit (close<static) qty=1.00 sl=976.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-21 09:15:00 | 1036.40 | 2024-05-21 09:15:00 | 1050.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-02 12:45:00 | 1026.05 | 2024-07-23 12:15:00 | 980.40 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2024-07-02 15:00:00 | 1023.95 | 2024-07-23 12:15:00 | 981.35 | PARTIAL | 0.50 | 4.16% |
| SELL | retest2 | 2024-07-02 12:45:00 | 1026.05 | 2024-07-23 14:15:00 | 1023.15 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2024-07-02 15:00:00 | 1023.95 | 2024-07-23 14:15:00 | 1023.15 | STOP_HIT | 0.50 | 0.08% |
| SELL | retest2 | 2024-07-03 09:45:00 | 1022.55 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-07-03 10:45:00 | 1025.35 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-07-04 10:15:00 | 1032.00 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-07-16 09:30:00 | 1033.00 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-07-24 10:15:00 | 1034.65 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-07-24 15:00:00 | 1035.20 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-07-25 11:15:00 | 1039.25 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-25 11:45:00 | 1036.25 | 2024-07-25 15:15:00 | 1057.80 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-08-12 10:15:00 | 1081.70 | 2024-08-28 11:15:00 | 1032.50 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2024-08-13 09:15:00 | 1114.15 | 2024-08-28 11:15:00 | 1032.50 | STOP_HIT | 1.00 | -7.33% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1081.95 | 2024-08-28 11:15:00 | 1032.50 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest2 | 2024-08-16 12:30:00 | 1080.40 | 2024-08-28 11:15:00 | 1032.50 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2024-09-25 10:00:00 | 1038.90 | 2024-10-04 09:15:00 | 986.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 14:00:00 | 1039.85 | 2024-10-04 09:15:00 | 987.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 09:30:00 | 1038.30 | 2024-10-04 09:15:00 | 986.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 13:00:00 | 1039.30 | 2024-10-04 09:15:00 | 987.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 13:15:00 | 1041.00 | 2024-10-04 09:15:00 | 988.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:45:00 | 1040.95 | 2024-10-04 09:15:00 | 988.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:45:00 | 1039.95 | 2024-10-04 09:15:00 | 987.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:45:00 | 1039.90 | 2024-10-04 09:15:00 | 987.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1032.90 | 2024-10-04 12:15:00 | 981.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 10:00:00 | 1038.90 | 2024-10-07 10:15:00 | 935.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-25 14:00:00 | 1039.85 | 2024-10-07 10:15:00 | 935.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 09:30:00 | 1038.30 | 2024-10-07 10:15:00 | 935.37 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2024-09-26 13:00:00 | 1039.30 | 2024-10-07 10:15:00 | 936.90 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2024-09-27 13:15:00 | 1041.00 | 2024-10-07 10:15:00 | 936.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-01 10:45:00 | 1040.95 | 2024-10-07 10:15:00 | 935.96 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2024-10-01 11:45:00 | 1039.95 | 2024-10-07 10:15:00 | 935.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-01 12:45:00 | 1039.90 | 2024-10-07 12:15:00 | 934.47 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1032.90 | 2024-10-07 13:15:00 | 929.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 12:15:00 | 1038.00 | 2024-10-23 09:15:00 | 986.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1036.80 | 2024-10-23 09:15:00 | 984.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 12:15:00 | 1038.00 | 2024-10-25 10:15:00 | 934.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1036.80 | 2024-10-25 10:15:00 | 933.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 09:15:00 | 993.85 | 2024-11-08 13:15:00 | 944.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:15:00 | 993.85 | 2024-11-13 14:15:00 | 894.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-16 10:45:00 | 791.90 | 2025-01-27 14:15:00 | 752.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 14:30:00 | 790.90 | 2025-01-27 14:15:00 | 751.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 11:45:00 | 791.35 | 2025-01-27 14:15:00 | 751.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:30:00 | 790.90 | 2025-01-27 14:15:00 | 751.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 10:45:00 | 791.90 | 2025-02-05 14:15:00 | 774.50 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-01-16 14:30:00 | 790.90 | 2025-02-05 14:15:00 | 774.50 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-01-17 11:45:00 | 791.35 | 2025-02-05 14:15:00 | 774.50 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2025-01-22 09:30:00 | 790.90 | 2025-02-05 14:15:00 | 774.50 | STOP_HIT | 0.50 | 2.07% |
| BUY | retest2 | 2025-04-07 10:45:00 | 811.30 | 2025-04-16 09:15:00 | 892.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 13:30:00 | 812.00 | 2025-04-16 09:15:00 | 893.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 14:30:00 | 811.65 | 2025-04-16 09:15:00 | 892.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 10:00:00 | 816.50 | 2025-05-13 09:15:00 | 898.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 858.20 | 2025-06-17 11:15:00 | 854.05 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-06-04 10:30:00 | 863.70 | 2025-06-17 11:15:00 | 854.05 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-13 12:30:00 | 859.45 | 2025-06-17 11:15:00 | 854.05 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-16 10:00:00 | 862.00 | 2025-06-17 11:15:00 | 854.05 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-06-27 09:30:00 | 883.10 | 2025-07-23 10:15:00 | 866.65 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-07-02 10:15:00 | 872.90 | 2025-07-23 10:15:00 | 866.65 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-03 11:45:00 | 871.85 | 2025-07-23 10:15:00 | 866.65 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-03 12:45:00 | 871.20 | 2025-07-23 10:15:00 | 866.65 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-14 10:15:00 | 885.90 | 2025-07-23 13:15:00 | 863.10 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-07-14 11:30:00 | 880.60 | 2025-07-23 13:15:00 | 863.10 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-07-14 14:15:00 | 878.55 | 2025-07-23 13:15:00 | 863.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-14 14:45:00 | 881.45 | 2025-07-23 13:15:00 | 863.10 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-07-21 10:15:00 | 879.30 | 2025-07-23 15:15:00 | 862.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-22 09:15:00 | 874.40 | 2025-07-23 15:15:00 | 862.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-22 10:00:00 | 874.00 | 2025-07-23 15:15:00 | 862.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-12 10:45:00 | 813.25 | 2025-09-12 11:15:00 | 819.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-03-05 15:15:00 | 989.70 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-03-06 11:00:00 | 991.00 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-03-10 09:15:00 | 992.50 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-03-10 10:00:00 | 989.90 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-03-12 10:15:00 | 997.50 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2026-03-13 11:15:00 | 998.60 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-13 12:45:00 | 998.40 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-03-13 14:00:00 | 997.00 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-03-16 13:15:00 | 984.20 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-03-17 10:00:00 | 985.60 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-17 10:45:00 | 984.30 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-03-17 13:15:00 | 985.70 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-17 14:45:00 | 1000.90 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest2 | 2026-04-06 14:30:00 | 995.10 | 2026-04-10 09:15:00 | 1094.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 995.90 | 2026-04-10 09:15:00 | 1095.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1037.60 | 2026-04-10 12:15:00 | 1141.36 | TARGET_HIT | 1.00 | 10.00% |
