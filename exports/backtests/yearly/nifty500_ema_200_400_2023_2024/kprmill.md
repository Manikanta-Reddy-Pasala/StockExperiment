# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 63 |
| PARTIAL | 8 |
| TARGET_HIT | 11 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 48
- **Target hits / Stop hits / Partials:** 11 / 52 / 8
- **Avg / median % per leg:** 0.00% / -1.90%
- **Sum % (uncompounded):** 0.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 8 | 23.5% | 8 | 26 | 0 | 0.25% | 8.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 8 | 23.5% | 8 | 26 | 0 | 0.25% | 8.6% |
| SELL (all) | 37 | 15 | 40.5% | 3 | 26 | 8 | -0.23% | -8.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 15 | 40.5% | 3 | 26 | 8 | -0.23% | -8.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 23 | 32.4% | 11 | 52 | 8 | 0.00% | 0.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 14:15:00 | 772.20 | 797.86 | 797.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-19 15:15:00 | 770.30 | 797.58 | 797.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 791.35 | 782.36 | 789.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 14:15:00 | 791.35 | 782.36 | 789.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 791.35 | 782.36 | 789.24 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 14:15:00 | 826.90 | 775.74 | 775.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 837.95 | 776.87 | 776.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 834.45 | 834.94 | 816.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:00:00 | 834.45 | 834.94 | 816.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 810.55 | 834.19 | 816.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 810.55 | 834.19 | 816.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 802.25 | 833.88 | 816.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 802.25 | 833.88 | 816.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 813.00 | 832.33 | 816.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 821.20 | 832.33 | 816.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 10:15:00 | 799.35 | 831.78 | 816.24 | SL hit (close<static) qty=1.00 sl=807.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 779.35 | 809.78 | 809.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 770.15 | 809.39 | 809.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 813.50 | 808.87 | 809.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 14:15:00 | 813.50 | 808.87 | 809.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 813.50 | 808.87 | 809.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 813.50 | 808.87 | 809.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 809.00 | 808.87 | 809.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 793.60 | 808.82 | 809.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 10:15:00 | 795.25 | 808.82 | 809.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 830.90 | 809.28 | 809.53 | SL hit (close>static) qty=1.00 sl=819.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 824.05 | 809.78 | 809.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 844.80 | 810.53 | 810.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 863.00 | 865.26 | 845.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 863.00 | 865.26 | 845.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 846.10 | 865.19 | 848.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 846.10 | 865.19 | 848.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 839.20 | 864.93 | 848.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 839.20 | 864.93 | 848.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 843.90 | 863.89 | 849.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 843.90 | 863.89 | 849.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 840.15 | 863.65 | 849.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:30:00 | 839.35 | 863.65 | 849.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 848.80 | 860.95 | 848.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 844.05 | 860.95 | 848.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 844.95 | 860.80 | 848.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:45:00 | 842.80 | 860.80 | 848.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 853.60 | 860.72 | 848.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 845.60 | 860.72 | 848.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 844.70 | 860.54 | 848.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 844.85 | 860.54 | 848.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 853.85 | 860.47 | 848.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 844.65 | 860.47 | 848.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 849.90 | 860.23 | 848.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:45:00 | 861.90 | 858.66 | 848.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:30:00 | 861.65 | 858.69 | 848.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:00:00 | 860.85 | 858.69 | 848.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:30:00 | 860.90 | 863.13 | 852.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 865.60 | 863.15 | 852.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 835.60 | 862.87 | 853.14 | SL hit (close<static) qty=1.00 sl=847.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 14:15:00 | 843.75 | 859.02 | 859.05 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 890.15 | 859.16 | 859.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 12:15:00 | 897.30 | 859.54 | 859.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 925.90 | 926.01 | 905.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 15:00:00 | 925.90 | 926.01 | 905.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 900.85 | 925.24 | 906.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 889.00 | 925.24 | 906.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 909.95 | 925.09 | 906.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:30:00 | 894.40 | 925.09 | 906.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 921.50 | 925.05 | 906.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 12:15:00 | 925.00 | 925.05 | 906.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 10:15:00 | 894.80 | 924.05 | 906.31 | SL hit (close<static) qty=1.00 sl=900.05 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 869.85 | 972.29 | 972.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 846.45 | 937.65 | 952.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 872.25 | 862.36 | 899.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-06 10:00:00 | 872.25 | 862.36 | 899.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 888.75 | 864.22 | 895.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:00:00 | 877.15 | 868.36 | 895.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 907.00 | 870.41 | 894.85 | SL hit (close>static) qty=1.00 sl=899.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 926.95 | 909.56 | 909.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 964.65 | 911.19 | 910.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1110.00 | 1116.81 | 1057.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 09:45:00 | 1114.30 | 1116.81 | 1057.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1078.30 | 1118.92 | 1076.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 1086.20 | 1118.20 | 1076.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1091.00 | 1113.75 | 1080.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 1084.40 | 1116.55 | 1089.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 1108.00 | 1116.16 | 1089.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 13:15:00 | 1192.84 | 1121.46 | 1094.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 987.50 | 1116.41 | 1116.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 975.60 | 1111.38 | 1114.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1023.20 | 1022.37 | 1051.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 1023.20 | 1022.37 | 1051.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1045.00 | 1023.25 | 1050.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1050.45 | 1023.25 | 1050.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1092.00 | 1024.20 | 1051.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1092.60 | 1024.20 | 1051.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1073.20 | 1024.69 | 1051.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1066.80 | 1058.43 | 1064.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 1072.55 | 1058.57 | 1064.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 1071.75 | 1058.72 | 1064.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 1114.20 | 1059.39 | 1064.28 | SL hit (close>static) qty=1.00 sl=1092.95 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 1080.30 | 1057.11 | 1057.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 1092.50 | 1060.73 | 1058.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1063.90 | 1074.62 | 1067.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1060.40 | 1074.47 | 1067.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1060.40 | 1074.47 | 1067.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 984.60 | 1061.45 | 1061.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 978.10 | 1060.62 | 1061.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 911.50 | 897.87 | 945.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:45:00 | 911.30 | 897.87 | 945.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 924.15 | 898.36 | 945.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 907.85 | 899.78 | 944.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 862.46 | 898.59 | 942.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.30 | 894.28 | 935.52 | SL hit (close>ema200) qty=0.50 sl=894.28 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 925.00 | 891.32 | 891.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 10:15:00 | 934.85 | 892.46 | 891.89 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-09 09:15:00 | 821.20 | 2024-05-09 10:15:00 | 799.35 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-05-14 10:15:00 | 830.00 | 2024-05-21 15:15:00 | 811.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-05-15 09:15:00 | 825.00 | 2024-05-21 15:15:00 | 811.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-15 09:45:00 | 820.95 | 2024-05-21 15:15:00 | 811.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-05-15 12:00:00 | 830.45 | 2024-05-21 15:15:00 | 811.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-05-15 13:45:00 | 830.55 | 2024-05-22 14:15:00 | 806.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-05-16 09:15:00 | 839.35 | 2024-05-22 14:15:00 | 806.00 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2024-05-17 15:15:00 | 841.00 | 2024-05-22 14:15:00 | 806.00 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2024-06-05 09:30:00 | 793.60 | 2024-06-06 09:15:00 | 830.90 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2024-06-05 10:15:00 | 795.25 | 2024-06-06 09:15:00 | 830.90 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2024-07-26 10:45:00 | 861.90 | 2024-08-05 10:15:00 | 835.60 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-07-26 11:30:00 | 861.65 | 2024-08-05 10:15:00 | 835.60 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-07-26 12:00:00 | 860.85 | 2024-08-05 10:15:00 | 835.60 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-08-01 14:30:00 | 860.90 | 2024-08-05 10:15:00 | 835.60 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-08-06 09:45:00 | 885.65 | 2024-08-07 09:15:00 | 974.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 09:45:00 | 868.05 | 2024-08-28 14:15:00 | 855.95 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-14 10:30:00 | 869.50 | 2024-08-28 15:15:00 | 855.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-08-16 09:15:00 | 871.60 | 2024-08-28 15:15:00 | 855.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-08-20 15:00:00 | 872.95 | 2024-08-28 15:15:00 | 855.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-08-21 10:15:00 | 870.75 | 2024-08-28 15:15:00 | 855.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-08-21 14:00:00 | 869.95 | 2024-08-29 11:15:00 | 836.55 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2024-08-21 14:30:00 | 868.90 | 2024-08-29 11:15:00 | 836.55 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-08-28 09:15:00 | 883.95 | 2024-08-29 11:15:00 | 836.55 | STOP_HIT | 1.00 | -5.36% |
| BUY | retest2 | 2024-09-06 09:15:00 | 873.05 | 2024-09-06 10:15:00 | 862.15 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-10-23 12:15:00 | 925.00 | 2024-10-24 10:15:00 | 894.80 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-10-31 09:15:00 | 926.55 | 2024-11-05 15:15:00 | 897.00 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-11-05 09:45:00 | 928.75 | 2024-11-05 15:15:00 | 897.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2024-11-05 11:30:00 | 921.70 | 2024-11-05 15:15:00 | 897.00 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-11-14 09:45:00 | 915.40 | 2024-12-03 09:15:00 | 1006.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-18 11:45:00 | 906.00 | 2024-12-03 09:15:00 | 996.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-18 15:15:00 | 905.00 | 2024-12-03 09:15:00 | 995.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-27 11:00:00 | 901.05 | 2025-01-27 13:15:00 | 877.75 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-03-17 12:00:00 | 877.15 | 2025-03-19 10:15:00 | 907.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-06-16 11:30:00 | 1086.20 | 2025-07-04 13:15:00 | 1192.84 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1091.00 | 2025-07-04 15:15:00 | 1194.82 | TARGET_HIT | 1.00 | 9.52% |
| BUY | retest2 | 2025-07-02 13:15:00 | 1084.40 | 2025-07-08 09:15:00 | 1200.10 | TARGET_HIT | 1.00 | 10.67% |
| BUY | retest2 | 2025-07-02 14:45:00 | 1108.00 | 2025-07-08 09:15:00 | 1218.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1066.80 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-09-26 10:00:00 | 1072.55 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-09-26 11:15:00 | 1071.75 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1071.50 | 2025-10-03 14:15:00 | 1072.40 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1054.40 | 2025-10-08 09:15:00 | 1017.92 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1056.80 | 2025-10-09 10:15:00 | 1007.09 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1060.10 | 2025-10-09 12:15:00 | 1003.96 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-10-07 10:00:00 | 1056.40 | 2025-10-09 12:15:00 | 1003.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1054.40 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1056.80 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1060.10 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-10-07 10:00:00 | 1056.40 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2025-10-17 14:15:00 | 1033.90 | 2025-10-23 09:15:00 | 1075.00 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-10-20 09:15:00 | 1028.80 | 2025-10-23 09:15:00 | 1075.00 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-10-24 13:45:00 | 1036.20 | 2025-10-27 14:15:00 | 1062.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-10-27 09:30:00 | 1035.00 | 2025-10-28 09:15:00 | 1066.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-27 12:15:00 | 1046.60 | 2025-10-28 09:15:00 | 1066.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1040.50 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-10-29 11:15:00 | 1045.80 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-29 12:00:00 | 1045.00 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1041.00 | 2025-11-11 14:15:00 | 1079.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-11-11 09:45:00 | 1025.60 | 2025-11-11 14:15:00 | 1079.00 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2026-01-29 09:15:00 | 907.85 | 2026-01-29 14:15:00 | 862.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 907.85 | 2026-02-03 09:15:00 | 1001.30 | STOP_HIT | 0.50 | -10.29% |
| SELL | retest2 | 2026-02-11 14:00:00 | 913.05 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -7.80% |
| SELL | retest2 | 2026-02-12 14:30:00 | 902.75 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest2 | 2026-02-16 09:30:00 | 910.00 | 2026-03-02 09:15:00 | 864.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:45:00 | 919.50 | 2026-03-02 09:15:00 | 873.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 915.00 | 2026-03-02 09:15:00 | 869.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 09:30:00 | 910.00 | 2026-03-05 13:15:00 | 827.55 | TARGET_HIT | 0.50 | 9.06% |
| SELL | retest2 | 2026-02-23 13:45:00 | 919.50 | 2026-03-09 09:15:00 | 819.00 | TARGET_HIT | 0.50 | 10.93% |
| SELL | retest2 | 2026-02-25 15:15:00 | 915.00 | 2026-03-09 09:15:00 | 823.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-17 15:15:00 | 922.00 | 2026-04-29 14:15:00 | 925.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-20 09:30:00 | 922.80 | 2026-04-29 14:15:00 | 925.00 | STOP_HIT | 1.00 | -0.24% |
