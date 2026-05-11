# The Ramco Cements Ltd. (RAMCOCEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 953.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 2 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 54 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 17 / 45
- **Target hits / Stop hits / Partials:** 3 / 50 / 9
- **Avg / median % per leg:** -0.47% / -1.65%
- **Sum % (uncompounded):** -28.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 5 | 19.2% | 3 | 22 | 1 | -0.06% | -1.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.09% | 6.2% |
| BUY @ 3rd Alert (retest2) | 24 | 3 | 12.5% | 3 | 21 | 0 | -0.32% | -7.7% |
| SELL (all) | 36 | 12 | 33.3% | 0 | 28 | 8 | -0.76% | -27.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.88% | -5.8% |
| SELL @ 3rd Alert (retest2) | 34 | 12 | 35.3% | 0 | 26 | 8 | -0.64% | -21.7% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.11% | 0.4% |
| retest2 (combined) | 58 | 15 | 25.9% | 3 | 47 | 8 | -0.51% | -29.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 12:15:00 | 955.90 | 984.83 | 984.89 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 1017.15 | 985.04 | 984.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 14:15:00 | 1018.80 | 985.97 | 985.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 14:15:00 | 985.55 | 987.07 | 986.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 14:15:00 | 985.55 | 987.07 | 986.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 985.55 | 987.07 | 986.01 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 903.20 | 984.82 | 985.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 13:15:00 | 897.35 | 983.95 | 984.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 12:15:00 | 839.80 | 837.37 | 877.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-12 09:15:00 | 846.40 | 840.93 | 870.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 789.75 | 773.80 | 799.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:30:00 | 781.35 | 774.21 | 799.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 821.15 | 774.80 | 799.45 | SL hit (close>static) qty=1.00 sl=800.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 13:15:00 | 854.00 | 817.15 | 817.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 11:15:00 | 857.80 | 818.72 | 817.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 820.95 | 831.35 | 825.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 10:15:00 | 820.95 | 831.35 | 825.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 820.95 | 831.35 | 825.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 820.95 | 831.35 | 825.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 823.10 | 831.27 | 825.34 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 790.05 | 820.60 | 820.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 784.50 | 816.31 | 818.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 15:15:00 | 810.00 | 806.13 | 812.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:15:00 | 802.90 | 806.13 | 812.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 818.45 | 806.26 | 812.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 818.45 | 806.26 | 812.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 817.60 | 806.37 | 812.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:30:00 | 813.70 | 806.54 | 812.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 12:15:00 | 826.15 | 806.74 | 812.52 | SL hit (close>static) qty=1.00 sl=821.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 835.95 | 814.14 | 814.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 843.40 | 814.97 | 814.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 828.25 | 831.25 | 824.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 11:00:00 | 828.25 | 831.25 | 824.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 822.70 | 831.16 | 824.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 820.25 | 831.16 | 824.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 819.90 | 831.05 | 824.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 819.90 | 831.05 | 824.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 822.75 | 830.97 | 824.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:45:00 | 820.90 | 830.97 | 824.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 845.85 | 853.88 | 842.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:00:00 | 853.35 | 853.80 | 842.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 836.05 | 853.37 | 842.82 | SL hit (close<static) qty=1.00 sl=839.05 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 863.15 | 933.00 | 933.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 852.00 | 902.50 | 914.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 10:15:00 | 874.00 | 869.33 | 889.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-06 10:30:00 | 870.35 | 869.33 | 889.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 876.25 | 869.58 | 889.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:30:00 | 886.60 | 869.58 | 889.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 865.10 | 857.29 | 875.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 864.95 | 857.90 | 875.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 863.85 | 858.07 | 875.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:00:00 | 862.60 | 858.12 | 875.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 907.60 | 859.56 | 874.91 | SL hit (close>static) qty=1.00 sl=888.55 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 948.00 | 886.40 | 886.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 955.15 | 891.54 | 888.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 935.20 | 937.19 | 919.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 09:15:00 | 950.00 | 937.23 | 920.29 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 13:15:00 | 997.50 | 943.26 | 924.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 961.25 | 965.64 | 943.36 | SL hit (close<ema200) qty=0.50 sl=965.64 alert=retest1 |

### Cycle 9 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 1045.00 | 1075.08 | 1075.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1024.10 | 1063.97 | 1068.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1030.40 | 1027.25 | 1044.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:15:00 | 1014.25 | 1027.45 | 1043.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1013.85 | 1027.07 | 1043.06 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1043.25 | 1026.88 | 1042.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1043.25 | 1026.88 | 1042.33 | SL hit (close>ema400) qty=1.00 sl=1042.33 alert=retest1 |

### Cycle 10 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 1061.40 | 1030.42 | 1030.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1074.70 | 1033.96 | 1032.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1057.50 | 1059.04 | 1048.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:45:00 | 1055.70 | 1059.04 | 1048.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1068.00 | 1059.86 | 1049.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 10:00:00 | 1074.60 | 1060.00 | 1049.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1071.00 | 1060.11 | 1049.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 12:00:00 | 1077.20 | 1060.28 | 1049.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 1046.00 | 1062.54 | 1051.87 | SL hit (close<static) qty=1.00 sl=1046.50 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 978.80 | 1084.18 | 1084.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 946.70 | 1078.76 | 1081.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 987.00 | 985.57 | 1022.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:30:00 | 985.60 | 985.57 | 1022.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1003.20 | 986.87 | 1017.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 996.30 | 987.18 | 1017.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 998.00 | 987.55 | 1017.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 998.00 | 991.95 | 1016.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:45:00 | 995.70 | 992.03 | 1016.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 946.48 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 948.10 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 948.10 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 945.91 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-06 14:30:00 | 781.35 | 2024-06-07 09:15:00 | 821.15 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2024-07-26 11:30:00 | 813.70 | 2024-07-26 12:15:00 | 826.15 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-05 09:15:00 | 813.75 | 2024-08-06 09:15:00 | 823.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-05 11:00:00 | 812.45 | 2024-08-06 09:15:00 | 823.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-06 14:00:00 | 812.25 | 2024-08-07 09:15:00 | 821.85 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-08-08 13:15:00 | 804.00 | 2024-08-19 09:15:00 | 817.30 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-08-08 13:45:00 | 805.45 | 2024-08-19 09:15:00 | 817.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-10-18 12:00:00 | 853.35 | 2024-10-21 11:15:00 | 836.05 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-10-24 12:30:00 | 853.20 | 2024-10-25 09:15:00 | 839.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-10-25 15:00:00 | 852.45 | 2024-11-22 11:15:00 | 937.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-28 10:45:00 | 854.00 | 2024-11-22 11:15:00 | 939.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-07 09:15:00 | 959.95 | 2025-01-07 12:15:00 | 949.95 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-07 10:00:00 | 957.00 | 2025-01-07 12:15:00 | 949.95 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-01-07 11:00:00 | 956.75 | 2025-01-07 12:15:00 | 949.95 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-01-07 12:15:00 | 956.00 | 2025-01-07 12:15:00 | 949.95 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-03-25 15:15:00 | 864.95 | 2025-03-28 09:15:00 | 907.60 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2025-03-26 09:30:00 | 863.85 | 2025-03-28 09:15:00 | 907.60 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2025-03-26 11:00:00 | 862.60 | 2025-03-28 09:15:00 | 907.60 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest1 | 2025-05-12 09:15:00 | 950.00 | 2025-05-14 13:15:00 | 997.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-12 09:15:00 | 950.00 | 2025-05-28 09:15:00 | 961.25 | STOP_HIT | 0.50 | 1.18% |
| BUY | retest2 | 2025-08-18 09:30:00 | 1085.50 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-08-18 10:00:00 | 1089.90 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-08-18 15:15:00 | 1082.80 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-08-19 13:15:00 | 1084.00 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-08-22 13:30:00 | 1077.90 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-08-22 15:15:00 | 1079.90 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-25 09:30:00 | 1080.00 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-09-02 11:00:00 | 1077.60 | 2025-09-02 13:15:00 | 1069.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1086.00 | 2025-09-05 15:15:00 | 1066.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-04 13:30:00 | 1086.10 | 2025-09-05 15:15:00 | 1066.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest1 | 2025-10-17 09:15:00 | 1014.25 | 2025-10-21 13:15:00 | 1043.25 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest1 | 2025-10-20 09:15:00 | 1013.85 | 2025-10-21 13:15:00 | 1043.25 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-11-03 11:45:00 | 1029.90 | 2025-11-14 09:15:00 | 978.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1030.50 | 2025-11-14 09:15:00 | 978.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1027.00 | 2025-11-14 09:15:00 | 975.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:45:00 | 1024.00 | 2025-11-14 09:15:00 | 972.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 11:45:00 | 1029.90 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1030.50 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1027.00 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 0.99% |
| SELL | retest2 | 2025-11-06 09:45:00 | 1024.00 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 0.70% |
| SELL | retest2 | 2025-11-21 15:00:00 | 1009.90 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-11-24 09:30:00 | 1012.50 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-11-24 13:30:00 | 1014.00 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-11-26 13:15:00 | 1014.40 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1025.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-01 14:30:00 | 1025.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1019.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-12-03 14:45:00 | 1025.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-04 10:00:00 | 1011.80 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-12-04 11:45:00 | 1011.50 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-12-04 13:45:00 | 1012.20 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1011.30 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2026-01-21 10:00:00 | 1074.60 | 2026-01-23 15:15:00 | 1046.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-01-21 11:00:00 | 1071.00 | 2026-01-23 15:15:00 | 1046.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-21 12:00:00 | 1077.20 | 2026-01-23 15:15:00 | 1046.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-01-27 15:00:00 | 1073.20 | 2026-02-09 09:15:00 | 1180.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-02 13:45:00 | 1105.70 | 2026-03-04 09:15:00 | 1066.30 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2026-03-05 15:15:00 | 1106.90 | 2026-03-06 10:15:00 | 1081.20 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-04-16 11:45:00 | 996.30 | 2026-04-28 15:15:00 | 946.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 15:15:00 | 998.00 | 2026-04-28 15:15:00 | 948.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 15:15:00 | 998.00 | 2026-04-28 15:15:00 | 948.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:45:00 | 995.70 | 2026-04-28 15:15:00 | 945.91 | PARTIAL | 0.50 | 5.00% |
