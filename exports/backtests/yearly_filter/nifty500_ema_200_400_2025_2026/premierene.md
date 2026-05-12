# Premier Energies Ltd. (PREMIERENE)

## Backtest Summary

- **Window:** 2024-09-03 09:15:00 → 2026-05-11 15:15:00 (2909 bars)
- **Last close:** 1002.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 24
- **Target hits / Stop hits / Partials:** 2 / 26 / 2
- **Avg / median % per leg:** -1.33% / -2.28%
- **Sum % (uncompounded):** -39.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -3.07% | -24.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -3.07% | -24.5% |
| SELL (all) | 22 | 6 | 27.3% | 2 | 18 | 2 | -0.69% | -15.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 2 | 18 | 2 | -0.69% | -15.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 6 | 20.0% | 2 | 26 | 2 | -1.33% | -39.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 1121.10 | 980.00 | 979.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 15:15:00 | 1135.00 | 981.54 | 980.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1026.40 | 1042.22 | 1020.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1026.40 | 1042.22 | 1020.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1026.40 | 1042.22 | 1020.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:15:00 | 1064.00 | 1029.95 | 1020.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:45:00 | 1060.30 | 1031.20 | 1020.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:30:00 | 1063.40 | 1031.81 | 1021.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:30:00 | 1069.40 | 1032.15 | 1021.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1051.90 | 1058.57 | 1042.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:30:00 | 1044.00 | 1058.57 | 1042.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1043.10 | 1058.12 | 1042.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 1043.10 | 1058.12 | 1042.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1043.60 | 1057.97 | 1042.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:45:00 | 1046.70 | 1057.78 | 1042.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 1054.10 | 1057.67 | 1042.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 10:15:00 | 1039.00 | 1062.47 | 1048.04 | SL hit (close<static) qty=1.00 sl=1041.20 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1013.50 | 1037.51 | 1037.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 994.50 | 1036.03 | 1036.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1029.80 | 1028.85 | 1032.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 1029.80 | 1028.85 | 1032.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1029.80 | 1028.85 | 1032.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 1032.00 | 1028.85 | 1032.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1037.20 | 1028.93 | 1032.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 1039.60 | 1028.93 | 1032.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 1037.70 | 1029.02 | 1032.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 1034.30 | 1029.54 | 1033.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:45:00 | 1033.50 | 1029.59 | 1033.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 982.58 | 1026.12 | 1030.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 981.82 | 1026.12 | 1030.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 1031.20 | 1022.25 | 1028.34 | SL hit (close>ema200) qty=0.50 sl=1022.25 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1076.40 | 1030.51 | 1030.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1018.80 | 1031.15 | 1031.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1013.00 | 1028.95 | 1030.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1027.90 | 1027.76 | 1029.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 1027.90 | 1027.76 | 1029.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1027.90 | 1027.76 | 1029.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1027.90 | 1027.76 | 1029.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1029.90 | 1027.79 | 1029.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1027.50 | 1027.79 | 1029.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1025.10 | 1027.76 | 1029.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1018.40 | 1027.46 | 1029.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 1020.00 | 1027.38 | 1029.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:00:00 | 1020.00 | 1027.31 | 1029.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 1020.00 | 1027.15 | 1028.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1030.10 | 1027.11 | 1028.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 1030.10 | 1027.11 | 1028.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1032.00 | 1027.16 | 1028.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1052.50 | 1027.16 | 1028.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 1047.40 | 1027.36 | 1029.02 | SL hit (close>static) qty=1.00 sl=1035.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1071.00 | 1030.61 | 1030.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1096.40 | 1041.13 | 1036.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1053.00 | 1055.78 | 1045.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:45:00 | 1053.00 | 1055.78 | 1045.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1039.00 | 1055.39 | 1045.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1039.00 | 1055.39 | 1045.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1035.90 | 1055.20 | 1045.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1028.00 | 1055.20 | 1045.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1045.00 | 1044.67 | 1041.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:15:00 | 1053.00 | 1044.75 | 1041.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 1060.00 | 1045.66 | 1041.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 1032.50 | 1045.62 | 1041.83 | SL hit (close<static) qty=1.00 sl=1039.10 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 997.80 | 1038.08 | 1038.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 986.40 | 1037.57 | 1037.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 799.00 | 770.32 | 836.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:30:00 | 793.20 | 770.32 | 836.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 789.00 | 779.24 | 828.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 780.35 | 779.24 | 828.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 781.50 | 779.35 | 828.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-25 09:15:00 | 702.32 | 771.98 | 810.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 891.45 | 806.97 | 806.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 906.10 | 812.68 | 809.61 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-30 11:15:00 | 1064.00 | 2025-07-31 10:15:00 | 1039.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-06-30 14:45:00 | 1060.30 | 2025-07-31 10:15:00 | 1039.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-07-01 09:30:00 | 1063.40 | 2025-08-01 13:15:00 | 1017.00 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2025-07-01 10:30:00 | 1069.40 | 2025-08-01 13:15:00 | 1017.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2025-07-22 14:45:00 | 1046.70 | 2025-08-01 13:15:00 | 1017.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-07-23 10:00:00 | 1054.10 | 2025-08-01 13:15:00 | 1017.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-08-21 10:45:00 | 1034.30 | 2025-08-26 09:15:00 | 982.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:45:00 | 1033.50 | 2025-08-26 09:15:00 | 981.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 10:45:00 | 1034.30 | 2025-09-01 12:15:00 | 1031.20 | STOP_HIT | 0.50 | 0.30% |
| SELL | retest2 | 2025-08-21 13:45:00 | 1033.50 | 2025-09-01 12:15:00 | 1031.20 | STOP_HIT | 0.50 | 0.22% |
| SELL | retest2 | 2025-09-01 14:00:00 | 1033.35 | 2025-09-02 09:15:00 | 1046.45 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-01 15:15:00 | 1035.90 | 2025-09-02 09:15:00 | 1046.45 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-04 12:30:00 | 1016.25 | 2025-09-10 14:15:00 | 1040.95 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-09-10 14:00:00 | 1017.75 | 2025-09-10 14:15:00 | 1040.95 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1018.40 | 2025-10-14 09:15:00 | 1047.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-10-13 10:00:00 | 1020.00 | 2025-10-14 09:15:00 | 1047.40 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1020.00 | 2025-10-14 09:15:00 | 1047.40 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-10-13 13:15:00 | 1020.00 | 2025-10-14 09:15:00 | 1047.40 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-11-12 12:15:00 | 1053.00 | 2025-11-14 09:15:00 | 1032.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-11-13 14:45:00 | 1060.00 | 2025-11-14 09:15:00 | 1032.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-11 10:15:00 | 780.35 | 2026-02-25 09:15:00 | 702.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-11 12:15:00 | 781.50 | 2026-02-25 09:15:00 | 703.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-11 12:15:00 | 782.70 | 2026-03-13 10:15:00 | 797.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-11 14:00:00 | 775.40 | 2026-03-13 10:15:00 | 797.90 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2026-03-12 09:15:00 | 773.90 | 2026-03-16 13:15:00 | 786.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-13 09:45:00 | 776.25 | 2026-03-16 13:15:00 | 786.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-03-13 13:15:00 | 778.20 | 2026-03-16 14:15:00 | 796.55 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-13 15:15:00 | 778.80 | 2026-03-16 14:15:00 | 796.55 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-03-16 09:15:00 | 773.30 | 2026-03-18 09:15:00 | 834.00 | STOP_HIT | 1.00 | -7.85% |
| SELL | retest2 | 2026-03-16 10:30:00 | 774.45 | 2026-03-18 09:15:00 | 834.00 | STOP_HIT | 1.00 | -7.69% |
