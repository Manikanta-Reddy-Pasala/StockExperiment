# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1277.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 11 |
| PENDING | 37 |
| PENDING_CANCEL | 11 |
| ENTRY1 | 2 |
| ENTRY2 | 24 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 21
- **Target hits / Stop hits / Partials:** 0 / 26 / 5
- **Avg / median % per leg:** 3.23% / -1.78%
- **Sum % (uncompounded):** 100.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 10 | 38.5% | 0 | 21 | 5 | 4.43% | 115.1% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.56% | -1.6% |
| BUY @ 3rd Alert (retest2) | 25 | 10 | 40.0% | 0 | 20 | 5 | 4.67% | 116.7% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.97% | -14.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.53% | -3.5% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.82% | -11.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.55% | -5.1% |
| retest2 (combined) | 29 | 10 | 34.5% | 0 | 24 | 5 | 3.63% | 105.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 756.45 | 790.50 | 790.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 748.15 | 780.92 | 784.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 759.45 | 758.71 | 769.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 777.35 | 758.91 | 768.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 777.35 | 758.91 | 768.93 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-11-16 09:15:00 | 766.25 | 759.82 | 769.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-16 10:15:00 | 771.45 | 759.94 | 769.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-16 12:15:00 | 768.90 | 760.14 | 769.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-16 13:15:00 | 769.25 | 760.23 | 769.07 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-16 14:15:00 | 767.35 | 760.30 | 769.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-16 15:15:00 | 769.15 | 760.39 | 769.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-17 15:15:00 | 768.80 | 761.03 | 769.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 09:15:00 | 768.30 | 761.10 | 769.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2023-11-22 12:15:00 | 768.25 | 762.65 | 769.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 13:15:00 | 766.10 | 762.69 | 769.22 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 782.85 | 763.73 | 769.44 | SL hit (close>static) qty=1.00 sl=782.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 782.85 | 763.73 | 769.44 | SL hit (close>static) qty=1.00 sl=782.30 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 12:15:00 | 820.05 | 774.02 | 773.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 824.20 | 776.29 | 775.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 837.20 | 838.08 | 816.15 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-05 09:15:00 | 843.80 | 838.15 | 817.15 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 10:15:00 | 837.15 | 838.14 | 817.25 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 818.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 818.10 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-10 11:15:00 | 824.40 | 835.70 | 818.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 12:15:00 | 823.95 | 835.58 | 818.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 09:15:00 | 827.50 | 834.56 | 818.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 10:15:00 | 825.90 | 834.47 | 818.62 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 13:15:00 | 823.15 | 834.12 | 818.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 14:15:00 | 824.50 | 834.03 | 818.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-15 11:15:00 | 828.80 | 833.68 | 818.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 12:15:00 | 827.80 | 833.62 | 818.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 816.60 | 832.80 | 819.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.28 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.28 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.28 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 810.25 | 832.05 | 819.28 | SL hit (close<static) qty=1.00 sl=811.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-30 10:15:00 | 826.00 | 823.71 | 817.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 11:15:00 | 825.70 | 823.73 | 817.17 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 813.15 | 823.61 | 817.21 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-05 09:15:00 | 826.10 | 821.76 | 816.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-05 10:15:00 | 821.55 | 821.76 | 816.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-07 09:15:00 | 828.15 | 821.20 | 816.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 829.00 | 821.28 | 817.02 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-08 11:15:00 | 827.25 | 822.28 | 817.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:15:00 | 828.00 | 822.33 | 817.74 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 814.25 | 822.24 | 817.79 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 814.25 | 822.24 | 817.79 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-21 09:15:00 | 841.50 | 819.08 | 816.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 844.25 | 819.33 | 817.11 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 813.60 | 820.79 | 818.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-26 09:15:00 | 813.60 | 820.79 | 818.11 | SL hit (close<static) qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-01 09:15:00 | 830.80 | 817.82 | 816.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 10:15:00 | 826.40 | 817.91 | 816.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-04 13:15:00 | 827.95 | 818.94 | 817.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-04 14:15:00 | 823.80 | 818.99 | 817.54 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 806.70 | 818.82 | 817.52 | SL hit (close<static) qty=1.00 sl=811.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-07 09:15:00 | 836.70 | 818.34 | 817.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 10:15:00 | 832.85 | 818.48 | 817.39 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 12:15:00 | 831.10 | 819.47 | 817.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 13:15:00 | 830.60 | 819.58 | 818.01 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-13 10:15:00 | 810.70 | 819.59 | 818.10 | SL hit (close<static) qty=1.00 sl=811.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 10:15:00 | 810.70 | 819.59 | 818.10 | SL hit (close<static) qty=1.00 sl=811.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 13:15:00 | 783.20 | 816.59 | 816.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 11:15:00 | 779.90 | 814.96 | 815.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.55 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-21 11:15:00 | 806.85 | 810.87 | 813.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-21 12:15:00 | 808.50 | 810.85 | 813.47 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 870.15 | 815.92 | 815.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 880.15 | 821.19 | 818.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 839.90 | 843.41 | 832.60 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-04-19 12:15:00 | 850.95 | 843.49 | 832.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 13:15:00 | 856.50 | 843.62 | 832.87 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 853.15 | 863.54 | 847.90 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-05-07 13:15:00 | 860.50 | 863.42 | 847.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-07 14:15:00 | 858.20 | 863.37 | 848.04 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-08 09:15:00 | 865.55 | 863.31 | 848.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-08 10:15:00 | 857.00 | 863.25 | 848.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-08 11:15:00 | 863.95 | 863.26 | 848.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:15:00 | 862.20 | 863.25 | 848.36 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-09 12:15:00 | 843.10 | 862.73 | 848.61 | SL hit (close<ema400) qty=1.00 sl=848.61 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-09 12:15:00 | 843.10 | 862.73 | 848.61 | SL hit (close<static) qty=1.00 sl=845.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-13 13:15:00 | 859.50 | 860.59 | 848.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-13 14:15:00 | 858.30 | 860.56 | 848.56 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-13 15:15:00 | 860.00 | 860.56 | 848.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 874.05 | 860.69 | 848.74 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 834.30 | 885.04 | 868.57 | SL hit (close<static) qty=1.00 sl=845.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 869.70 | 882.32 | 867.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 879.90 | 882.29 | 867.96 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:15:00 | 1011.89 | 952.61 | 935.86 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 985.60 | 987.81 | 963.40 | SL hit (close<ema200) qty=0.50 sl=987.81 alert=retest2 |

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
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 930.65 | 1019.38 | 995.34 | SL hit (close<static) qty=1.00 sl=945.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-08 13:15:00 | 958.00 | 1010.68 | 992.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-08 14:15:00 | 953.70 | 1010.11 | 991.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 983.35 | 1004.96 | 990.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 986.90 | 1004.78 | 990.10 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 963.00 | 1012.28 | 1000.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 964.10 | 1011.81 | 1000.56 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 958.20 | 1001.88 | 996.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 957.40 | 1001.44 | 996.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 957.40 | 1001.00 | 996.13 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 982.40 | 999.96 | 995.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 987.50 | 999.84 | 995.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:15:00 | 1108.71 | 1052.11 | 1041.36 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:15:00 | 1101.01 | 1052.11 | 1041.36 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 12:15:00 | 1134.94 | 1084.77 | 1064.15 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 12:15:00 | 1135.62 | 1084.77 | 1064.15 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.30 | 1137.37 | 1108.95 | SL hit (close<ema200) qty=0.50 sl=1137.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.30 | 1137.37 | 1108.95 | SL hit (close<ema200) qty=0.50 sl=1137.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.30 | 1137.37 | 1108.95 | SL hit (close<ema200) qty=0.50 sl=1137.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.30 | 1137.37 | 1108.95 | SL hit (close<ema200) qty=0.50 sl=1137.37 alert=retest2 |

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
| SELL | retest2 | 2023-11-20 09:15:00 | 768.30 | 2023-11-24 09:15:00 | 782.85 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-11-22 13:15:00 | 766.10 | 2023-11-24 09:15:00 | 782.85 | STOP_HIT | 1.00 | -2.19% |
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
| BUY | retest1 | 2024-04-19 13:15:00 | 856.50 | 2024-05-09 12:15:00 | 843.10 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-05-08 12:15:00 | 862.20 | 2024-05-09 12:15:00 | 843.10 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-05-14 09:15:00 | 874.05 | 2024-06-04 11:15:00 | 834.30 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2024-06-05 14:15:00 | 879.90 | 2024-09-27 09:15:00 | 1011.89 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 14:15:00 | 879.90 | 2024-10-16 11:15:00 | 985.60 | STOP_HIT | 0.50 | 12.01% |
| SELL | retest1 | 2025-01-22 12:15:00 | 913.90 | 2025-01-24 10:15:00 | 946.15 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-01-27 10:15:00 | 911.40 | 2025-01-30 09:15:00 | 954.00 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-02-03 10:15:00 | 925.75 | 2025-02-05 09:15:00 | 949.30 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-03-03 10:15:00 | 956.75 | 2025-04-07 09:15:00 | 930.65 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-04-11 10:15:00 | 986.90 | 2025-09-08 09:15:00 | 1108.71 | PARTIAL | 0.50 | 12.34% |
| BUY | retest2 | 2025-05-06 10:15:00 | 964.10 | 2025-09-08 09:15:00 | 1101.01 | PARTIAL | 0.50 | 14.20% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2025-09-23 12:15:00 | 1134.94 | PARTIAL | 0.50 | 18.54% |
| BUY | retest2 | 2025-05-12 10:15:00 | 987.50 | 2025-09-23 12:15:00 | 1135.62 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-11 10:15:00 | 986.90 | 2025-10-23 14:15:00 | 1136.30 | STOP_HIT | 0.50 | 15.14% |
| BUY | retest2 | 2025-05-06 10:15:00 | 964.10 | 2025-10-23 14:15:00 | 1136.30 | STOP_HIT | 0.50 | 17.86% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2025-10-23 14:15:00 | 1136.30 | STOP_HIT | 0.50 | 18.69% |
| BUY | retest2 | 2025-05-12 10:15:00 | 987.50 | 2025-10-23 14:15:00 | 1136.30 | STOP_HIT | 0.50 | 15.07% |
| BUY | retest2 | 2026-03-05 10:15:00 | 1239.40 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -4.26% |
