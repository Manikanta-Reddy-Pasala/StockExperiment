# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 780.85
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
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 12 |
| PENDING | 34 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 21 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 22
- **Target hits / Stop hits / Partials:** 3 / 24 / 1
- **Avg / median % per leg:** -0.31% / -0.95%
- **Sum % (uncompounded):** -8.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 4 | 25.0% | 2 | 14 | 0 | -0.08% | -1.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.37% | -5.4% |
| BUY @ 3rd Alert (retest2) | 15 | 4 | 26.7% | 2 | 13 | 0 | 0.27% | 4.1% |
| SELL (all) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.60% | -7.3% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.13% | 0.8% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.34% | -8.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.65% | -4.6% |
| retest2 (combined) | 21 | 4 | 19.0% | 2 | 19 | 0 | -0.19% | -4.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 12:15:00 | 826.00 | 774.31 | 774.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 15:15:00 | 828.45 | 789.69 | 782.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 822.83 | 824.57 | 808.75 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-15 09:15:00 | 832.50 | 824.54 | 809.51 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:15:00 | 835.45 | 824.65 | 809.64 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 790.62 | 825.64 | 811.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 790.62 | 825.64 | 811.10 | SL hit (close<ema400) qty=1.00 sl=811.10 alert=retest1 |

### Cycle 2 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 716.92 | 798.86 | 799.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 713.60 | 798.01 | 798.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 724.83 | 724.31 | 745.17 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-11 09:15:00 | 714.75 | 723.97 | 743.58 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-11 10:15:00 | 717.28 | 723.90 | 743.45 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 714.67 | 723.81 | 743.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 713.50 | 723.71 | 743.16 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-20 10:15:00 | 714.65 | 724.21 | 739.38 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-20 11:15:00 | 718.50 | 724.15 | 739.28 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-20 14:15:00 | 714.70 | 723.97 | 738.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 15:15:00 | 716.00 | 723.89 | 738.85 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-26 09:15:00 | 716.20 | 723.59 | 737.62 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:15:00 | 715.65 | 723.51 | 737.51 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 733.15 | 723.04 | 735.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-02 09:15:00 | 744.62 | 723.86 | 735.88 | SL hit (close>ema400) qty=1.00 sl=735.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-02 09:15:00 | 744.62 | 723.86 | 735.88 | SL hit (close>ema400) qty=1.00 sl=735.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-02 09:15:00 | 744.62 | 723.86 | 735.88 | SL hit (close>ema400) qty=1.00 sl=735.88 alert=retest1 |

### Cycle 3 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 759.75 | 744.00 | 743.95 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 730.47 | 745.93 | 745.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 725.97 | 745.44 | 745.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 743.88 | 740.11 | 742.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 743.88 | 740.11 | 742.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 743.88 | 740.11 | 742.71 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 754.62 | 745.02 | 744.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 11:15:00 | 757.83 | 745.52 | 745.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 741.83 | 749.76 | 747.49 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 767.95 | 750.08 | 747.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 776.05 | 750.34 | 747.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-06-25 12:15:00 | 853.65 | 785.15 | 768.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 10:15:00 | 826.45 | 876.38 | 876.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 11:15:00 | 821.00 | 875.83 | 876.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 12:15:00 | 851.33 | 848.52 | 859.07 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 839.03 | 848.44 | 858.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 10:15:00 | 842.95 | 848.39 | 858.75 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 858.15 | 848.08 | 858.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 858.15 | 848.08 | 858.13 | SL hit (close>ema400) qty=1.00 sl=858.13 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-11 09:15:00 | 849.45 | 852.38 | 858.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-11 10:15:00 | 851.90 | 852.37 | 858.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 12:15:00 | 848.22 | 852.34 | 858.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:15:00 | 847.05 | 852.28 | 858.80 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-12 09:15:00 | 843.88 | 852.17 | 858.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 10:15:00 | 842.03 | 852.07 | 858.56 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-13 13:15:00 | 848.53 | 852.10 | 858.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 849.03 | 852.07 | 858.22 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 859.08 | 851.56 | 857.54 | SL hit (close>static) qty=1.00 sl=858.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 859.08 | 851.56 | 857.54 | SL hit (close>static) qty=1.00 sl=858.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 859.08 | 851.56 | 857.54 | SL hit (close>static) qty=1.00 sl=858.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 09:15:00 | 845.03 | 852.97 | 857.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 10:15:00 | 845.35 | 852.90 | 857.75 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 857.50 | 850.25 | 855.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-28 11:15:00 | 858.28 | 850.33 | 855.52 | SL hit (close>static) qty=1.00 sl=858.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-03 13:15:00 | 848.55 | 850.92 | 855.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 14:15:00 | 850.60 | 850.92 | 855.57 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-04 09:15:00 | 849.83 | 850.92 | 855.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-04 10:15:00 | 852.20 | 850.93 | 855.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-05 10:15:00 | 850.53 | 851.13 | 855.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:15:00 | 850.70 | 851.13 | 855.43 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-12 12:15:00 | 858.65 | 849.59 | 853.87 | SL hit (close>static) qty=1.00 sl=857.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-12 12:15:00 | 858.65 | 849.59 | 853.87 | SL hit (close>static) qty=1.00 sl=857.88 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 897.05 | 857.20 | 857.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 898.95 | 857.62 | 857.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.77 | 869.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 12:15:00 | 872.08 | 878.65 | 869.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 872.08 | 878.65 | 869.72 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-07 14:15:00 | 879.30 | 878.60 | 869.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 878.85 | 878.61 | 869.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-04-21 09:15:00 | 966.73 | 892.89 | 879.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 958.75 | 980.49 | 980.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 947.40 | 971.45 | 974.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 968.95 | 965.15 | 970.88 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 957.75 | 965.04 | 970.68 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 10:15:00 | 962.90 | 965.02 | 970.64 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.48 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1002.50 | 974.02 | 973.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.36 | 974.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.97 | 981.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 979.75 | 986.89 | 981.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 979.75 | 986.89 | 981.79 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-06 11:15:00 | 987.60 | 986.84 | 981.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 989.10 | 986.86 | 981.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 15:15:00 | 984.25 | 986.40 | 981.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 986.35 | 986.40 | 981.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 984.10 | 986.30 | 982.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 986.85 | 986.30 | 982.06 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-14 11:15:00 | 984.90 | 986.72 | 982.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 985.85 | 986.71 | 982.71 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 983.35 | 986.68 | 982.71 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-14 14:15:00 | 990.50 | 986.72 | 982.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 989.00 | 986.74 | 982.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-19 11:15:00 | 987.30 | 987.62 | 983.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 988.30 | 987.63 | 983.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 13:15:00 | 988.50 | 995.84 | 990.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 989.40 | 995.77 | 990.51 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.68 | 991.29 | SL hit (close<static) qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.68 | 991.29 | SL hit (close<static) qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 981.30 | 995.68 | 991.29 | SL hit (close<static) qty=1.00 sl=981.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 988.30 | 993.74 | 990.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 987.30 | 993.68 | 990.63 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 990.20 | 993.65 | 990.94 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-26 14:15:00 | 992.30 | 993.60 | 990.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:15:00 | 992.20 | 993.59 | 990.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 988.30 | 993.50 | 990.94 | SL hit (close<static) qty=1.00 sl=988.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-29 14:15:00 | 992.20 | 993.42 | 990.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-29 15:15:00 | 991.40 | 993.40 | 990.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-31 12:15:00 | 995.00 | 992.97 | 990.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 13:15:00 | 996.40 | 993.00 | 990.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-31 15:15:00 | 994.00 | 992.99 | 990.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 994.10 | 993.00 | 990.91 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-01-01 12:15:00 | 992.40 | 992.97 | 990.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 992.85 | 992.97 | 990.94 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 990.60 | 992.95 | 990.94 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-02 09:15:00 | 996.30 | 992.96 | 990.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 996.00 | 992.99 | 990.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 988.00 | 993.26 | 991.20 | SL hit (close<static) qty=1.00 sl=988.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 988.00 | 993.26 | 991.20 | SL hit (close<static) qty=1.00 sl=988.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 988.00 | 993.26 | 991.20 | SL hit (close<static) qty=1.00 sl=988.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 988.00 | 993.26 | 991.20 | SL hit (close<static) qty=1.00 sl=990.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 978.15 | 992.91 | 991.05 | SL hit (close<static) qty=1.00 sl=981.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.32 | 990.78 | SL hit (close<static) qty=1.00 sl=974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.32 | 990.78 | SL hit (close<static) qty=1.00 sl=974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.32 | 990.78 | SL hit (close<static) qty=1.00 sl=974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 963.25 | 992.32 | 990.78 | SL hit (close<static) qty=1.00 sl=974.20 alert=retest2 |

### Cycle 10 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.03 | 989.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.44 | 988.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 950.75 | 948.17 | 963.09 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-03 15:15:00 | 947.20 | 948.25 | 962.76 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-04 09:15:00 | 953.10 | 948.30 | 962.72 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 944.15 | 948.62 | 961.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:15:00 | 942.50 | 948.56 | 961.81 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 895.38 | 929.26 | 944.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-09 09:15:00 | 848.25 | 912.50 | 933.34 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-15 10:15:00 | 835.45 | 2024-01-17 09:15:00 | 790.62 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest1 | 2024-03-11 12:15:00 | 713.50 | 2024-04-02 09:15:00 | 744.62 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest1 | 2024-03-20 15:15:00 | 716.00 | 2024-04-02 09:15:00 | 744.62 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest1 | 2024-03-26 10:15:00 | 715.65 | 2024-04-02 09:15:00 | 744.62 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-06-05 14:15:00 | 776.05 | 2024-06-25 12:15:00 | 853.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-02-03 10:15:00 | 842.95 | 2025-02-04 12:15:00 | 858.15 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-02-11 13:15:00 | 847.05 | 2025-02-17 14:15:00 | 859.08 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-02-12 10:15:00 | 842.03 | 2025-02-17 14:15:00 | 859.08 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-02-13 14:15:00 | 849.03 | 2025-02-17 14:15:00 | 859.08 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-02-20 10:15:00 | 845.35 | 2025-02-28 11:15:00 | 858.28 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-03-03 14:15:00 | 850.60 | 2025-03-12 12:15:00 | 858.65 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-03-05 11:15:00 | 850.70 | 2025-03-12 12:15:00 | 858.65 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-04-07 15:15:00 | 878.85 | 2025-04-21 09:15:00 | 966.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-06 12:15:00 | 989.10 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-10 09:15:00 | 986.35 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-11 12:15:00 | 986.85 | 2025-12-17 12:15:00 | 981.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-14 12:15:00 | 985.85 | 2025-12-29 11:15:00 | 988.30 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-11-14 15:15:00 | 989.00 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-11-19 12:15:00 | 988.30 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-12-10 14:15:00 | 989.40 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-12-22 10:15:00 | 987.30 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-12-26 15:15:00 | 992.20 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-31 13:15:00 | 996.40 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-01-01 09:15:00 | 994.10 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-01-01 13:15:00 | 992.85 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-01-02 10:15:00 | 996.00 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest1 | 2026-02-06 10:15:00 | 942.50 | 2026-02-27 09:15:00 | 895.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-06 10:15:00 | 942.50 | 2026-03-09 09:15:00 | 848.25 | TARGET_HIT | 0.50 | 10.00% |
