# JSWSTEEL (JSWSTEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 1273.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 32 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 1 |
| ENTRY2 | 23 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 21
- **Target hits / Stop hits / Partials:** 0 / 24 / 3
- **Avg / median % per leg:** 2.10% / -1.55%
- **Sum % (uncompounded):** 56.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 6 | 22.2% | 0 | 24 | 3 | 2.10% | 56.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.01% | -1.0% |
| BUY @ 3rd Alert (retest2) | 26 | 6 | 23.1% | 0 | 23 | 3 | 2.22% | 57.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.01% | -1.0% |
| retest2 (combined) | 26 | 6 | 23.1% | 0 | 23 | 3 | 2.22% | 57.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 10:15:00 | 818.15 | 773.07 | 772.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 11:15:00 | 821.65 | 773.55 | 773.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 837.20 | 838.08 | 815.89 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-05 09:15:00 | 843.80 | 838.14 | 816.90 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 10:15:00 | 837.15 | 838.13 | 817.00 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 817.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 815.70 | 835.95 | 817.87 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-01-10 11:15:00 | 824.40 | 835.70 | 817.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 12:15:00 | 823.95 | 835.58 | 817.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 09:15:00 | 827.50 | 834.56 | 818.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 10:15:00 | 825.90 | 834.47 | 818.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 13:15:00 | 823.15 | 834.12 | 818.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 14:15:00 | 824.50 | 834.03 | 818.51 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-15 11:15:00 | 828.80 | 833.67 | 818.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 12:15:00 | 827.80 | 833.62 | 818.68 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 816.60 | 832.80 | 819.20 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 811.20 | 832.27 | 819.14 | SL hit qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 811.20 | 832.27 | 819.14 | SL hit qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 811.20 | 832.27 | 819.14 | SL hit qty=1.00 sl=811.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 811.20 | 832.27 | 819.14 | SL hit qty=1.00 sl=811.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-30 10:15:00 | 826.00 | 823.71 | 816.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 11:15:00 | 825.70 | 823.73 | 817.02 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 815.40 | 823.61 | 817.06 | SL hit qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-05 09:15:00 | 826.10 | 821.76 | 816.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-05 10:15:00 | 821.55 | 821.76 | 816.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-07 09:15:00 | 828.15 | 821.20 | 816.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 829.00 | 821.28 | 816.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-08 11:15:00 | 827.25 | 822.27 | 817.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:15:00 | 828.00 | 822.33 | 817.63 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 815.40 | 822.24 | 817.68 | SL hit qty=1.00 sl=815.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 815.40 | 822.24 | 817.68 | SL hit qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-21 09:15:00 | 841.50 | 819.08 | 816.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 844.25 | 819.33 | 817.02 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 813.60 | 820.79 | 818.03 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-02-26 09:15:00 | 815.40 | 820.79 | 818.03 | SL hit qty=1.00 sl=815.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-01 09:15:00 | 830.80 | 817.82 | 816.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 10:15:00 | 826.40 | 817.91 | 816.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-04 13:15:00 | 827.95 | 818.94 | 817.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-04 14:15:00 | 823.80 | 818.99 | 817.48 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-03-05 11:15:00 | 811.10 | 818.98 | 817.50 | SL hit qty=1.00 sl=811.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-07 09:15:00 | 836.70 | 818.34 | 817.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 10:15:00 | 832.85 | 818.48 | 817.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 12:15:00 | 831.10 | 819.47 | 817.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 13:15:00 | 830.60 | 819.58 | 817.96 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 811.10 | 819.68 | 818.09 | SL hit qty=1.00 sl=811.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 811.10 | 819.68 | 818.09 | SL hit qty=1.00 sl=811.10 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 13:15:00 | 783.20 | 816.59 | 816.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 11:15:00 | 779.90 | 814.96 | 815.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.51 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 812.45 | 810.92 | 813.51 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-03-21 11:15:00 | 806.85 | 810.87 | 813.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-21 12:15:00 | 808.50 | 810.85 | 813.43 | ENTRY2 sustain failed after 60m |

### Cycle 3 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 870.15 | 815.92 | 815.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 880.15 | 821.19 | 818.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 839.90 | 843.41 | 832.58 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-04-19 12:15:00 | 850.95 | 843.49 | 832.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 13:15:00 | 856.50 | 843.62 | 832.84 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 853.15 | 863.54 | 847.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-07 11:15:00 | 847.88 | 863.54 | 847.88 | SL hit qty=1.00 sl=847.88 alert=retest1 |
| Cross detected — sustain check pending | 2024-05-07 13:15:00 | 860.50 | 863.42 | 847.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-07 14:15:00 | 858.20 | 863.37 | 848.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-08 09:15:00 | 865.55 | 863.31 | 848.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-08 10:15:00 | 857.00 | 863.25 | 848.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-08 11:15:00 | 863.95 | 863.26 | 848.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:15:00 | 862.20 | 863.25 | 848.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-09 12:15:00 | 845.90 | 862.73 | 848.60 | SL hit qty=1.00 sl=845.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-13 13:15:00 | 859.50 | 860.59 | 848.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-13 14:15:00 | 858.30 | 860.56 | 848.55 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-13 15:15:00 | 860.00 | 860.56 | 848.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 874.05 | 860.69 | 848.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 845.90 | 885.04 | 868.56 | SL hit qty=1.00 sl=845.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 869.70 | 882.32 | 867.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 879.90 | 882.29 | 867.95 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-09-27 09:15:00 | 1011.89 | 952.61 | 935.86 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-12-24 15:15:00 | 921.85 | 967.88 | 968.10 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-02-17 11:15:00 | 961.85 | 942.99 | 942.97 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 11:15:00 | 961.85 | 942.99 | 942.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 966.85 | 943.44 | 943.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 955.00 | 955.30 | 950.05 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 12:15:00 | 945.90 | 955.18 | 950.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 945.90 | 955.18 | 950.04 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-03 09:15:00 | 958.10 | 955.01 | 950.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 956.75 | 955.03 | 950.09 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 945.00 | 1019.38 | 995.34 | SL hit qty=1.00 sl=945.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-08 13:15:00 | 958.00 | 1010.68 | 992.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-08 14:15:00 | 953.70 | 1010.11 | 991.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 983.35 | 1004.96 | 990.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 986.90 | 1004.78 | 990.10 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 963.00 | 1012.28 | 1000.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 964.10 | 1011.81 | 1000.56 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 945.00 | 1002.88 | 996.96 | SL hit qty=1.00 sl=945.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 945.00 | 1002.88 | 996.96 | SL hit qty=1.00 sl=945.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 958.20 | 1001.88 | 996.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 957.40 | 1001.44 | 996.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 957.40 | 1001.00 | 996.13 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 982.40 | 999.96 | 995.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 987.50 | 999.84 | 995.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-08 09:15:00 | 1101.01 | 1052.11 | 1041.36 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-23 12:15:00 | 1135.62 | 1084.77 | 1064.15 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-12-18 09:15:00 | 1076.90 | 1134.90 | 1135.07 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1189.00 | 1132.07 | 1131.79 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1189.00 | 1132.07 | 1131.79 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1189.00 | 1132.07 | 1131.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1213.50 | 1159.18 | 1148.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.40 | 1231.46 | 1203.72 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 1202.80 | 1231.04 | 1203.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1202.80 | 1231.04 | 1203.78 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-05 09:15:00 | 1230.10 | 1230.32 | 1204.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1239.40 | 1230.41 | 1204.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1202.80 | 1230.95 | 1206.20 | SL hit qty=1.00 sl=1202.80 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-24 11:15:00 | 1119.50 | 1190.49 | 1190.79 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1222.20 | 1172.88 | 1179.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1221.10 | 1173.36 | 1180.09 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1202.80 | 1175.47 | 1180.96 | SL hit qty=1.00 sl=1202.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 1222.40 | 1177.68 | 1181.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 1217.90 | 1178.08 | 1182.08 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-15 12:15:00 | 1220.50 | 1178.85 | 1182.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 1220.80 | 1179.27 | 1182.61 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1202.80 | 1183.09 | 1184.41 | SL hit qty=1.00 sl=1202.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1202.80 | 1183.09 | 1184.41 | SL hit qty=1.00 sl=1202.80 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 1241.70 | 1185.98 | 1185.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 1250.00 | 1186.62 | 1186.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-10 12:15:00 | 823.95 | 2024-01-17 14:15:00 | 811.20 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-01-12 10:15:00 | 825.90 | 2024-01-17 14:15:00 | 811.20 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-01-12 14:15:00 | 824.50 | 2024-01-17 14:15:00 | 811.20 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-01-15 12:15:00 | 827.80 | 2024-01-17 14:15:00 | 811.20 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-01-30 11:15:00 | 825.70 | 2024-01-30 14:15:00 | 815.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-02-07 10:15:00 | 829.00 | 2024-02-09 09:15:00 | 815.40 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-02-08 12:15:00 | 828.00 | 2024-02-09 09:15:00 | 815.40 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-02-21 10:15:00 | 844.25 | 2024-02-26 09:15:00 | 815.40 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2024-03-01 10:15:00 | 826.40 | 2024-03-05 11:15:00 | 811.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-03-07 10:15:00 | 832.85 | 2024-03-13 09:15:00 | 811.10 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-03-11 13:15:00 | 830.60 | 2024-03-13 09:15:00 | 811.10 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2024-04-19 13:15:00 | 856.50 | 2024-05-07 11:15:00 | 847.88 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-05-08 12:15:00 | 862.20 | 2024-05-09 12:15:00 | 845.90 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-05-14 09:15:00 | 874.05 | 2024-06-04 11:15:00 | 845.90 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2024-06-05 14:15:00 | 879.90 | 2024-09-27 09:15:00 | 1011.89 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 14:15:00 | 879.90 | 2025-02-17 11:15:00 | 961.85 | STOP_HIT | 0.50 | 9.31% |
| BUY | retest2 | 2025-03-03 10:15:00 | 956.75 | 2025-04-07 09:15:00 | 945.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-04-11 10:15:00 | 986.90 | 2025-05-09 09:15:00 | 945.00 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-05-06 10:15:00 | 964.10 | 2025-05-09 09:15:00 | 945.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2025-09-08 09:15:00 | 1101.01 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-05-12 10:15:00 | 987.50 | 2025-09-23 12:15:00 | 1135.62 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2026-01-06 09:15:00 | 1189.00 | STOP_HIT | 0.50 | 24.19% |
| BUY | retest2 | 2025-05-12 10:15:00 | 987.50 | 2026-01-06 09:15:00 | 1189.00 | STOP_HIT | 0.50 | 20.41% |
| BUY | retest2 | 2026-03-05 10:15:00 | 1239.40 | 2026-03-09 09:15:00 | 1202.80 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-04-10 10:15:00 | 1221.10 | 2026-04-13 09:15:00 | 1202.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-15 10:15:00 | 1217.90 | 2026-04-17 09:15:00 | 1202.80 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-04-15 13:15:00 | 1220.80 | 2026-04-17 09:15:00 | 1202.80 | STOP_HIT | 1.00 | -1.47% |
