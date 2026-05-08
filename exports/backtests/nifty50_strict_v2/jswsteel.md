# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1277.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 13 |
| PENDING | 45 |
| PENDING_CANCEL | 10 |
| ENTRY1 | 2 |
| ENTRY2 | 33 |
| PARTIAL | 1 |
| TARGET_HIT | 7 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 25
- **Target hits / Stop hits / Partials:** 7 / 28 / 1
- **Avg / median % per leg:** 0.15% / -1.78%
- **Sum % (uncompounded):** 5.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 11 | 36.7% | 7 | 22 | 1 | 0.76% | 22.7% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 2.36% | 4.7% |
| BUY @ 3rd Alert (retest2) | 28 | 10 | 35.7% | 7 | 21 | 0 | 0.64% | 18.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.92% | -17.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.53% | -3.5% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.79% | -14.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.40% | 1.2% |
| retest2 (combined) | 33 | 10 | 30.3% | 7 | 26 | 0 | 0.12% | 4.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 760.30 | 791.78 | 791.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 755.80 | 791.11 | 791.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 785.40 | 782.28 | 786.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 785.40 | 782.28 | 786.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 785.40 | 782.28 | 786.55 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-10-13 09:15:00 | 774.60 | 782.33 | 786.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 10:15:00 | 775.55 | 782.27 | 786.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-16 12:15:00 | 793.85 | 782.27 | 786.19 | SL hit (close>static) qty=1.00 sl=789.95 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-19 09:15:00 | 775.90 | 783.42 | 786.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 10:15:00 | 778.45 | 783.37 | 786.43 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-20 09:15:00 | 772.55 | 783.05 | 786.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 10:15:00 | 770.50 | 782.92 | 786.09 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-29 13:15:00 | 791.45 | 767.03 | 770.83 | SL hit (close>static) qty=1.00 sl=789.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-29 13:15:00 | 791.45 | 767.03 | 770.83 | SL hit (close>static) qty=1.00 sl=789.95 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 13:15:00 | 820.20 | 774.47 | 774.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 824.20 | 776.28 | 775.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 837.20 | 838.08 | 816.22 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-05 09:15:00 | 843.80 | 838.14 | 817.21 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 10:15:00 | 837.15 | 838.13 | 817.31 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 818.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 818.16 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-10 11:15:00 | 824.40 | 835.70 | 818.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 12:15:00 | 823.95 | 835.58 | 818.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 09:15:00 | 827.50 | 834.56 | 818.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 10:15:00 | 825.90 | 834.47 | 818.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 13:15:00 | 823.15 | 834.12 | 818.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 14:15:00 | 824.50 | 834.03 | 818.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-15 11:15:00 | 828.80 | 833.67 | 818.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 12:15:00 | 827.80 | 833.62 | 818.93 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 816.60 | 832.80 | 819.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.33 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.33 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.33 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.33 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-30 10:15:00 | 826.00 | 823.71 | 817.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 11:15:00 | 825.70 | 823.73 | 817.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 813.15 | 823.61 | 817.25 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-05 09:15:00 | 826.10 | 821.76 | 816.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-05 10:15:00 | 821.55 | 821.76 | 816.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-07 09:15:00 | 828.15 | 821.20 | 816.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 829.00 | 821.28 | 817.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-08 11:15:00 | 827.25 | 822.27 | 817.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:15:00 | 828.00 | 822.33 | 817.77 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 814.25 | 822.24 | 817.82 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 814.25 | 822.24 | 817.82 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-21 09:15:00 | 841.50 | 819.08 | 816.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 844.25 | 819.33 | 817.13 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 813.60 | 820.79 | 818.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-26 09:15:00 | 813.60 | 820.79 | 818.13 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-01 09:15:00 | 830.80 | 817.82 | 816.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 10:15:00 | 826.40 | 817.91 | 816.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-04 13:15:00 | 827.95 | 818.94 | 817.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-04 14:15:00 | 823.80 | 818.99 | 817.56 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 806.70 | 818.82 | 817.53 | SL hit (close<static) qty=1.00 sl=811.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-07 09:15:00 | 836.70 | 818.34 | 817.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 10:15:00 | 832.85 | 818.48 | 817.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 12:15:00 | 831.10 | 819.47 | 817.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 13:15:00 | 830.60 | 819.58 | 818.02 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-13 10:15:00 | 810.70 | 819.59 | 818.12 | SL hit (close<static) qty=1.00 sl=811.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 10:15:00 | 810.70 | 819.59 | 818.12 | SL hit (close<static) qty=1.00 sl=811.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 13:15:00 | 783.20 | 816.59 | 816.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 11:15:00 | 779.90 | 814.96 | 815.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.56 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-21 11:15:00 | 806.85 | 810.87 | 813.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-21 12:15:00 | 808.50 | 810.85 | 813.48 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 870.15 | 815.92 | 815.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 880.15 | 821.19 | 818.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 839.90 | 843.41 | 832.61 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-04-19 12:15:00 | 850.95 | 843.49 | 832.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 13:15:00 | 856.50 | 843.62 | 832.87 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 14:15:00 | 899.33 | 850.42 | 837.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-07 10:15:00 | 854.10 | 863.65 | 847.87 | SL hit (close<ema200) qty=0.50 sl=863.65 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 853.15 | 863.54 | 847.90 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-05-07 13:15:00 | 860.50 | 863.42 | 847.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-07 14:15:00 | 858.20 | 863.37 | 848.04 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-08 09:15:00 | 865.55 | 863.31 | 848.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-08 10:15:00 | 857.00 | 863.25 | 848.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-08 11:15:00 | 863.95 | 863.26 | 848.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:15:00 | 862.20 | 863.25 | 848.36 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-09 12:15:00 | 843.10 | 862.73 | 848.61 | SL hit (close<static) qty=1.00 sl=845.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-13 13:15:00 | 859.50 | 860.59 | 848.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-13 14:15:00 | 858.30 | 860.56 | 848.56 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-13 15:15:00 | 860.00 | 860.56 | 848.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 874.05 | 860.69 | 848.75 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 834.30 | 885.04 | 868.57 | SL hit (close<static) qty=1.00 sl=845.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 869.70 | 882.32 | 867.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 879.90 | 882.29 | 867.96 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2024-08-26 14:15:00 | 967.89 | 914.32 | 908.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 921.85 | 967.88 | 968.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 919.75 | 966.54 | 967.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 924.80 | 923.60 | 939.07 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-22 11:15:00 | 915.35 | 923.73 | 938.46 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 12:15:00 | 913.90 | 923.64 | 938.34 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 932.45 | 923.99 | 937.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-24 10:15:00 | 946.15 | 924.21 | 937.77 | SL hit (close>ema400) qty=1.00 sl=937.77 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-27 09:15:00 | 918.05 | 924.79 | 937.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 911.40 | 924.66 | 937.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 954.00 | 924.63 | 936.26 | SL hit (close>static) qty=1.00 sl=945.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 925.85 | 927.52 | 936.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 925.75 | 927.50 | 936.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 949.30 | 928.81 | 937.00 | SL hit (close>static) qty=1.00 sl=945.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 11:15:00 | 961.85 | 942.99 | 942.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 966.85 | 943.44 | 943.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 955.00 | 955.30 | 950.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 12:15:00 | 945.90 | 955.18 | 950.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 945.90 | 955.18 | 950.04 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-03-03 09:15:00 | 958.10 | 955.01 | 950.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 956.75 | 955.03 | 950.09 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-03-20 13:15:00 | 1052.43 | 987.96 | 971.23 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-08 13:15:00 | 958.00 | 1010.68 | 992.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-08 14:15:00 | 953.70 | 1010.11 | 991.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 983.35 | 1004.96 | 990.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 986.90 | 1004.78 | 990.10 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 963.00 | 1012.28 | 1000.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 964.10 | 1011.81 | 1000.56 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 958.20 | 1001.88 | 996.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 957.40 | 1001.44 | 996.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 999.60 | 999.85 | 995.76 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-15 09:15:00 | 1017.90 | 999.30 | 995.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:15:00 | 1019.00 | 999.50 | 995.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 993.60 | 1004.90 | 999.43 | SL hit (close<static) qty=1.00 sl=995.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-23 09:15:00 | 1013.70 | 1004.74 | 999.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 1018.00 | 1004.87 | 999.63 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-26 12:15:00 | 1034.50 | 1005.33 | 1000.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 1032.50 | 1005.60 | 1000.26 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-29 09:15:00 | 1018.10 | 1006.82 | 1001.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-29 10:15:00 | 1007.50 | 1006.82 | 1001.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-29 12:15:00 | 1010.00 | 1006.81 | 1001.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-29 13:15:00 | 1005.00 | 1006.80 | 1001.44 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 992.20 | 1006.19 | 1001.34 | SL hit (close<static) qty=1.00 sl=995.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 992.20 | 1006.19 | 1001.34 | SL hit (close<static) qty=1.00 sl=995.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 10:15:00 | 1010.20 | 998.38 | 997.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1010.00 | 998.49 | 997.90 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 997.25 | 1000.20 | 998.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 989.00 | 1000.00 | 998.79 | SL hit (close<static) qty=1.00 sl=995.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 1002.95 | 999.12 | 998.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 1008.00 | 999.21 | 998.45 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-17 13:15:00 | 1001.00 | 999.45 | 998.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 1001.00 | 999.46 | 998.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 993.00 | 999.34 | 998.56 | SL hit (close<static) qty=1.00 sl=994.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 993.00 | 999.34 | 998.56 | SL hit (close<static) qty=1.00 sl=994.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 1000.40 | 998.47 | 998.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1022.00 | 998.70 | 998.27 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 987.90 | 999.09 | 998.48 | SL hit (close<static) qty=1.00 sl=994.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-23 12:15:00 | 1002.80 | 999.02 | 998.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 1001.90 | 999.04 | 998.47 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 997.20 | 999.03 | 998.46 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-24 10:15:00 | 1016.55 | 999.20 | 998.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 1018.30 | 999.39 | 998.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-07-02 11:15:00 | 1060.51 | 1008.26 | 1003.64 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-02 11:15:00 | 1053.14 | 1008.26 | 1003.64 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-20 12:15:00 | 1085.59 | 1046.16 | 1033.73 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-08 09:15:00 | 1102.09 | 1052.11 | 1041.36 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-18 13:15:00 | 1120.13 | 1076.46 | 1058.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1076.90 | 1134.90 | 1135.07 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1189.00 | 1132.07 | 1131.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1213.50 | 1159.18 | 1148.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.40 | 1231.46 | 1203.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 1202.80 | 1231.04 | 1203.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1202.80 | 1231.04 | 1203.78 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-05 09:15:00 | 1230.10 | 1230.32 | 1204.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1239.40 | 1230.41 | 1204.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1230.95 | 1206.20 | SL hit (close<static) qty=1.00 sl=1202.80 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 11:15:00 | 1119.50 | 1190.49 | 1190.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1117.30 | 1181.70 | 1186.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1186.40 | 1168.62 | 1178.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1186.40 | 1168.62 | 1178.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1186.40 | 1168.62 | 1178.35 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 1241.70 | 1185.98 | 1185.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 1250.00 | 1186.62 | 1186.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-13 10:15:00 | 775.55 | 2023-10-16 12:15:00 | 793.85 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2023-10-19 10:15:00 | 778.45 | 2023-11-29 13:15:00 | 791.45 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-10-20 10:15:00 | 770.50 | 2023-11-29 13:15:00 | 791.45 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-01-10 12:15:00 | 823.95 | 2024-01-17 15:15:00 | 810.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-01-12 10:15:00 | 825.90 | 2024-01-17 15:15:00 | 810.25 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-01-12 14:15:00 | 824.50 | 2024-01-17 15:15:00 | 810.25 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-01-15 12:15:00 | 827.80 | 2024-01-17 15:15:00 | 810.25 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-01-30 11:15:00 | 825.70 | 2024-01-30 14:15:00 | 813.15 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-02-07 10:15:00 | 829.00 | 2024-02-09 09:15:00 | 814.25 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-02-08 12:15:00 | 828.00 | 2024-02-09 09:15:00 | 814.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-02-21 10:15:00 | 844.25 | 2024-02-26 09:15:00 | 813.60 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-03-01 10:15:00 | 826.40 | 2024-03-06 09:15:00 | 806.70 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-03-07 10:15:00 | 832.85 | 2024-03-13 10:15:00 | 810.70 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-03-11 13:15:00 | 830.60 | 2024-03-13 10:15:00 | 810.70 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2024-04-19 13:15:00 | 856.50 | 2024-04-25 14:15:00 | 899.33 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-19 13:15:00 | 856.50 | 2024-05-07 10:15:00 | 854.10 | STOP_HIT | 0.50 | -0.28% |
| BUY | retest2 | 2024-05-08 12:15:00 | 862.20 | 2024-05-09 12:15:00 | 843.10 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-05-14 09:15:00 | 874.05 | 2024-06-04 11:15:00 | 834.30 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2024-06-05 14:15:00 | 879.90 | 2024-08-26 14:15:00 | 967.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-01-22 12:15:00 | 913.90 | 2025-01-24 10:15:00 | 946.15 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-01-27 10:15:00 | 911.40 | 2025-01-30 09:15:00 | 954.00 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-02-03 10:15:00 | 925.75 | 2025-02-05 09:15:00 | 949.30 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-03-03 10:15:00 | 956.75 | 2025-03-20 13:15:00 | 1052.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-11 10:15:00 | 986.90 | 2025-05-22 09:15:00 | 993.60 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2025-05-06 10:15:00 | 964.10 | 2025-05-30 14:15:00 | 992.20 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2025-05-30 14:15:00 | 992.20 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest2 | 2025-05-15 10:15:00 | 1019.00 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-05-23 10:15:00 | 1018.00 | 2025-06-18 10:15:00 | 993.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-05-26 13:15:00 | 1032.50 | 2025-06-18 10:15:00 | 993.00 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2025-06-09 11:15:00 | 1010.00 | 2025-06-23 09:15:00 | 987.90 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-16 13:15:00 | 1008.00 | 2025-07-02 11:15:00 | 1060.51 | TARGET_HIT | 1.00 | 5.21% |
| BUY | retest2 | 2025-06-17 14:15:00 | 1001.00 | 2025-07-02 11:15:00 | 1053.14 | TARGET_HIT | 1.00 | 5.21% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1022.00 | 2025-08-20 12:15:00 | 1085.59 | TARGET_HIT | 1.00 | 6.22% |
| BUY | retest2 | 2025-06-23 13:15:00 | 1001.90 | 2025-09-08 09:15:00 | 1102.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 11:15:00 | 1018.30 | 2025-09-18 13:15:00 | 1120.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-05 10:15:00 | 1239.40 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -4.26% |
