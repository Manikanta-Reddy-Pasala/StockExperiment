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
| CROSSOVER | 233 |
| ALERT1 | 146 |
| ALERT2 | 144 |
| ALERT2_SKIP | 95 |
| ALERT3 | 355 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 173 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 176 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 184 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 143
- **Target hits / Stop hits / Partials:** 0 / 176 / 8
- **Avg / median % per leg:** -0.30% / -0.85%
- **Sum % (uncompounded):** -54.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 97 | 23 | 23.7% | 0 | 97 | 0 | -0.38% | -37.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 97 | 23 | 23.7% | 0 | 97 | 0 | -0.38% | -37.2% |
| SELL (all) | 87 | 18 | 20.7% | 0 | 79 | 8 | -0.20% | -17.4% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.52% | 6.1% |
| SELL @ 3rd Alert (retest2) | 83 | 15 | 18.1% | 0 | 76 | 7 | -0.28% | -23.5% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.52% | 6.1% |
| retest2 (combined) | 180 | 38 | 21.1% | 0 | 173 | 7 | -0.34% | -60.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 772.35 | 771.36 | 771.34 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 12:15:00 | 769.05 | 771.16 | 771.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 15:15:00 | 767.10 | 769.70 | 770.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-16 11:15:00 | 770.25 | 768.27 | 769.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 11:15:00 | 770.25 | 768.27 | 769.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 11:15:00 | 770.25 | 768.27 | 769.51 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 15:15:00 | 771.00 | 770.25 | 770.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 09:15:00 | 779.40 | 772.08 | 771.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 11:15:00 | 772.55 | 774.12 | 772.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 11:15:00 | 772.55 | 774.12 | 772.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 772.55 | 774.12 | 772.28 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 902.00 | 914.68 | 915.72 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 09:15:00 | 928.00 | 916.69 | 916.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 11:15:00 | 932.00 | 921.50 | 918.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 13:15:00 | 935.15 | 936.57 | 932.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 14:15:00 | 933.85 | 936.03 | 932.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 933.85 | 936.03 | 932.58 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 935.15 | 939.74 | 940.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 11:15:00 | 930.00 | 935.47 | 937.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 932.00 | 931.31 | 934.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 932.00 | 931.31 | 934.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 932.00 | 931.31 | 934.53 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 945.95 | 937.33 | 936.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 14:15:00 | 949.25 | 939.71 | 937.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 931.75 | 939.18 | 937.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 931.75 | 939.18 | 937.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 931.75 | 939.18 | 937.88 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 928.45 | 935.63 | 936.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 09:15:00 | 922.50 | 930.79 | 932.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 935.20 | 928.58 | 930.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 935.20 | 928.58 | 930.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 935.20 | 928.58 | 930.59 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 944.80 | 933.16 | 932.41 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 929.00 | 932.13 | 932.47 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 936.85 | 933.08 | 932.87 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 927.60 | 933.21 | 933.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 12:15:00 | 924.00 | 931.36 | 932.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 931.50 | 928.48 | 930.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 931.50 | 928.48 | 930.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 931.50 | 928.48 | 930.53 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 12:15:00 | 938.65 | 932.79 | 932.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 14:15:00 | 942.20 | 935.47 | 933.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 13:15:00 | 938.35 | 938.64 | 936.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 14:15:00 | 936.00 | 938.11 | 936.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 936.00 | 938.11 | 936.32 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 929.65 | 937.48 | 938.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 14:15:00 | 926.95 | 931.62 | 934.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 10:15:00 | 930.10 | 930.02 | 933.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 10:15:00 | 930.10 | 930.02 | 933.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 930.10 | 930.02 | 933.13 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 936.35 | 930.67 | 930.40 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 13:15:00 | 924.70 | 930.06 | 930.59 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 943.55 | 931.29 | 930.74 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 928.00 | 930.81 | 930.95 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 09:15:00 | 936.45 | 931.94 | 931.45 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 926.65 | 930.82 | 931.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 925.15 | 929.69 | 930.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 928.10 | 927.90 | 929.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 928.10 | 927.90 | 929.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 928.10 | 927.90 | 929.24 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 13:15:00 | 895.45 | 894.60 | 894.50 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 15:15:00 | 887.35 | 893.07 | 893.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 11:15:00 | 880.95 | 889.38 | 891.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 880.00 | 879.73 | 883.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 892.85 | 882.69 | 884.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 892.85 | 882.69 | 884.45 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 894.70 | 886.46 | 885.94 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 13:15:00 | 883.50 | 885.62 | 885.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 14:15:00 | 881.20 | 884.74 | 885.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 12:15:00 | 884.90 | 882.65 | 883.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 12:15:00 | 884.90 | 882.65 | 883.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 884.90 | 882.65 | 883.92 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 868.40 | 860.39 | 859.54 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 855.15 | 858.74 | 859.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 15:15:00 | 854.50 | 857.89 | 858.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 13:15:00 | 859.95 | 856.21 | 857.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 13:15:00 | 859.95 | 856.21 | 857.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 859.95 | 856.21 | 857.39 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 14:15:00 | 852.95 | 843.08 | 842.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 11:15:00 | 855.05 | 848.54 | 845.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 09:15:00 | 851.50 | 852.72 | 849.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 11:15:00 | 848.40 | 853.77 | 852.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 848.40 | 853.77 | 852.20 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 850.00 | 851.32 | 851.33 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 858.50 | 852.76 | 851.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 10:15:00 | 860.40 | 854.29 | 852.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 15:15:00 | 858.00 | 858.22 | 855.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 15:15:00 | 858.00 | 858.22 | 855.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 858.00 | 858.22 | 855.61 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 850.10 | 854.90 | 855.06 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 857.00 | 855.32 | 855.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 10:15:00 | 860.00 | 856.26 | 855.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 10:15:00 | 865.45 | 868.99 | 866.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 10:15:00 | 865.45 | 868.99 | 866.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 865.45 | 868.99 | 866.17 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 866.30 | 867.09 | 867.13 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 870.95 | 867.87 | 867.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 12:15:00 | 878.45 | 870.47 | 868.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 15:15:00 | 915.50 | 917.44 | 906.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 15:15:00 | 910.00 | 914.02 | 910.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 910.00 | 914.02 | 910.05 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 13:15:00 | 900.60 | 908.31 | 908.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 09:15:00 | 895.45 | 903.57 | 906.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 11:15:00 | 902.65 | 901.59 | 904.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 14:15:00 | 902.70 | 901.69 | 903.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 902.70 | 901.69 | 903.96 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 888.30 | 873.11 | 872.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 902.30 | 878.95 | 875.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 11:15:00 | 920.50 | 922.63 | 909.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 10:15:00 | 912.85 | 920.24 | 914.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 912.85 | 920.24 | 914.24 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 928.65 | 938.30 | 939.48 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 14:15:00 | 951.05 | 940.85 | 940.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 966.25 | 950.84 | 946.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 995.95 | 997.34 | 987.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 1010.00 | 998.87 | 989.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1010.00 | 998.87 | 989.96 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 986.90 | 991.40 | 991.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 986.00 | 990.32 | 991.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 15:15:00 | 990.00 | 988.66 | 990.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 09:15:00 | 983.55 | 987.64 | 989.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 983.55 | 987.64 | 989.51 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 1001.70 | 989.62 | 989.29 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 12:15:00 | 986.00 | 990.32 | 990.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 13:15:00 | 984.00 | 989.05 | 989.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 988.75 | 985.08 | 987.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 988.75 | 985.08 | 987.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 988.75 | 985.08 | 987.55 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 13:15:00 | 983.90 | 978.72 | 978.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 15:15:00 | 986.00 | 981.11 | 979.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 989.20 | 990.86 | 987.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 13:15:00 | 986.05 | 989.90 | 986.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 986.05 | 989.90 | 986.92 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 986.95 | 988.39 | 988.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 15:15:00 | 983.00 | 987.31 | 987.93 | Break + close below crossover candle low |

### Cycle 43 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 1002.70 | 990.39 | 989.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 10:15:00 | 1010.45 | 994.40 | 991.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 1002.00 | 1004.53 | 999.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 12:15:00 | 1002.00 | 1004.53 | 999.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 1002.00 | 1004.53 | 999.96 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 11:15:00 | 980.50 | 1001.87 | 1003.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 13:15:00 | 980.35 | 994.62 | 999.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 1010.80 | 994.70 | 997.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 09:15:00 | 1010.80 | 994.70 | 997.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 1010.80 | 994.70 | 997.71 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1005.50 | 998.54 | 998.53 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 12:15:00 | 996.00 | 998.48 | 998.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-15 15:15:00 | 994.00 | 997.17 | 997.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 12:15:00 | 996.90 | 994.53 | 996.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 12:15:00 | 996.90 | 994.53 | 996.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 996.90 | 994.53 | 996.20 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-11-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 14:15:00 | 1003.75 | 997.55 | 997.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 09:15:00 | 1015.00 | 1001.43 | 999.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 15:15:00 | 1004.40 | 1005.72 | 1002.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 995.10 | 1003.60 | 1002.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 995.10 | 1003.60 | 1002.17 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 988.00 | 998.62 | 1000.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 14:15:00 | 984.90 | 992.20 | 996.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 10:15:00 | 990.90 | 989.90 | 994.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 968.50 | 965.13 | 967.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 968.50 | 965.13 | 967.91 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 986.30 | 970.75 | 970.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 11:15:00 | 991.60 | 977.09 | 973.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 12:15:00 | 988.55 | 990.04 | 983.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 12:15:00 | 1017.05 | 1023.95 | 1019.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 1017.05 | 1023.95 | 1019.68 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 1012.00 | 1021.28 | 1022.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 1008.00 | 1018.62 | 1020.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 14:15:00 | 1017.00 | 1016.92 | 1019.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 15:15:00 | 1020.75 | 1017.68 | 1019.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 1020.75 | 1017.68 | 1019.75 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 1029.45 | 1022.47 | 1021.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 1032.05 | 1025.21 | 1023.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 13:15:00 | 1046.75 | 1048.59 | 1042.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 14:15:00 | 1040.45 | 1046.96 | 1042.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 1040.45 | 1046.96 | 1042.70 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 12:15:00 | 1034.15 | 1041.38 | 1041.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 13:15:00 | 1033.00 | 1039.70 | 1040.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 10:15:00 | 1019.85 | 1017.05 | 1024.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 11:15:00 | 1023.25 | 1018.29 | 1024.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 1023.25 | 1018.29 | 1024.63 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 1025.85 | 995.65 | 994.51 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 1000.90 | 1014.11 | 1014.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 994.25 | 1010.13 | 1012.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 13:15:00 | 1013.50 | 1009.51 | 1011.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 13:15:00 | 1013.50 | 1009.51 | 1011.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 1013.50 | 1009.51 | 1011.76 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 09:15:00 | 1015.80 | 1013.06 | 1013.04 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 10:15:00 | 1011.40 | 1012.73 | 1012.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 12:15:00 | 1008.55 | 1011.66 | 1012.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 1013.20 | 1009.95 | 1011.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 1013.20 | 1009.95 | 1011.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1013.20 | 1009.95 | 1011.11 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 1030.60 | 1014.80 | 1012.67 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 13:15:00 | 1004.55 | 1014.98 | 1015.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 996.10 | 1011.21 | 1014.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 11:15:00 | 989.65 | 979.81 | 987.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 11:15:00 | 989.65 | 979.81 | 987.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 989.65 | 979.81 | 987.50 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 13:15:00 | 995.00 | 988.93 | 988.84 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 10:15:00 | 983.95 | 988.34 | 988.75 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 994.55 | 989.21 | 989.05 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 14:15:00 | 985.90 | 988.86 | 988.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 15:15:00 | 982.60 | 987.61 | 988.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 990.00 | 988.09 | 988.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 990.00 | 988.09 | 988.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 990.00 | 988.09 | 988.51 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 991.80 | 988.83 | 988.81 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 986.05 | 988.27 | 988.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 983.25 | 987.27 | 988.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 13:15:00 | 992.70 | 988.36 | 988.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 13:15:00 | 992.70 | 988.36 | 988.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 992.70 | 988.36 | 988.50 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 14:15:00 | 989.85 | 988.65 | 988.62 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 15:15:00 | 987.05 | 988.33 | 988.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 976.10 | 985.89 | 987.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 964.10 | 955.54 | 962.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 964.10 | 955.54 | 962.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 964.10 | 955.54 | 962.33 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 964.20 | 956.64 | 955.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 14:15:00 | 968.15 | 959.90 | 958.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 984.15 | 985.02 | 977.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 1009.00 | 989.29 | 980.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1009.00 | 989.29 | 980.81 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 09:15:00 | 988.80 | 995.08 | 995.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 980.80 | 990.00 | 992.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 996.00 | 989.36 | 991.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 996.00 | 989.36 | 991.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 996.00 | 989.36 | 991.42 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 13:15:00 | 998.45 | 993.72 | 993.09 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 986.10 | 992.94 | 993.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 12:15:00 | 981.65 | 989.47 | 991.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 15:15:00 | 861.45 | 861.19 | 876.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 875.50 | 866.68 | 871.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 875.50 | 866.68 | 871.26 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 879.95 | 872.19 | 871.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 13:15:00 | 882.95 | 876.14 | 873.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 10:15:00 | 875.55 | 877.70 | 875.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 11:15:00 | 883.05 | 878.77 | 876.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 883.05 | 878.77 | 876.14 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 870.00 | 877.16 | 877.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 869.85 | 875.70 | 876.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 869.85 | 869.45 | 872.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 871.85 | 869.81 | 872.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 871.85 | 869.81 | 872.13 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 12:15:00 | 872.30 | 872.10 | 872.08 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 09:15:00 | 869.55 | 871.62 | 871.87 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 874.80 | 872.25 | 872.14 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 11:15:00 | 870.95 | 871.99 | 872.03 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 12:15:00 | 872.40 | 872.07 | 872.06 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 869.60 | 871.58 | 871.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 856.40 | 867.68 | 869.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 15:15:00 | 843.80 | 839.50 | 848.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 848.15 | 841.23 | 848.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 848.15 | 841.23 | 848.49 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 805.00 | 802.46 | 802.12 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 10:15:00 | 799.00 | 801.80 | 801.89 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 802.60 | 801.96 | 801.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 14:15:00 | 805.25 | 803.01 | 802.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 15:15:00 | 801.35 | 802.68 | 802.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 15:15:00 | 801.35 | 802.68 | 802.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 801.35 | 802.68 | 802.37 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 794.00 | 800.94 | 801.61 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 809.35 | 800.20 | 799.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 810.00 | 802.16 | 800.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 807.30 | 810.87 | 807.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 14:15:00 | 807.30 | 810.87 | 807.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 807.30 | 810.87 | 807.63 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 09:15:00 | 840.05 | 843.55 | 843.93 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 847.90 | 842.75 | 842.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 851.35 | 845.52 | 844.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 848.00 | 848.23 | 846.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 12:15:00 | 840.50 | 846.42 | 845.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 840.50 | 846.42 | 845.61 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 838.00 | 844.74 | 844.92 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 847.60 | 844.95 | 844.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 13:15:00 | 850.00 | 846.57 | 845.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 843.80 | 846.56 | 845.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 843.80 | 846.56 | 845.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 843.80 | 846.56 | 845.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 842.95 | 846.56 | 845.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 846.10 | 846.47 | 845.88 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 842.25 | 845.34 | 845.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 839.10 | 844.09 | 844.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 15:15:00 | 812.90 | 811.27 | 819.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 09:15:00 | 806.60 | 811.27 | 819.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 801.70 | 799.52 | 804.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 12:30:00 | 799.00 | 799.06 | 803.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 13:15:00 | 812.05 | 802.06 | 802.08 | SL hit (close>static) qty=1.00 sl=809.95 alert=retest2 |

### Cycle 89 — BUY (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 14:15:00 | 810.40 | 803.73 | 802.84 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 10:15:00 | 790.00 | 804.76 | 805.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 11:15:00 | 783.50 | 800.51 | 803.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 09:15:00 | 797.10 | 794.00 | 798.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 09:15:00 | 797.10 | 794.00 | 798.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 797.10 | 794.00 | 798.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:45:00 | 797.05 | 794.00 | 798.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 801.40 | 795.48 | 798.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:00:00 | 801.40 | 795.48 | 798.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 802.35 | 796.85 | 798.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:30:00 | 802.85 | 796.85 | 798.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 796.65 | 798.31 | 799.21 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 11:15:00 | 801.60 | 799.66 | 799.62 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 13:15:00 | 798.00 | 799.29 | 799.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 794.95 | 797.79 | 798.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 783.00 | 782.53 | 788.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 783.00 | 782.53 | 788.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 783.00 | 782.53 | 788.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:45:00 | 783.10 | 782.53 | 788.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 767.90 | 763.52 | 767.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 767.90 | 763.52 | 767.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 764.20 | 763.66 | 767.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:30:00 | 757.85 | 762.93 | 766.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:30:00 | 761.90 | 762.79 | 765.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 15:15:00 | 763.95 | 759.00 | 758.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 763.95 | 759.00 | 758.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 765.35 | 760.61 | 759.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 764.50 | 765.36 | 762.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 764.50 | 765.36 | 762.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 762.45 | 764.78 | 762.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 761.20 | 764.78 | 762.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 763.10 | 764.44 | 762.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:45:00 | 760.90 | 764.44 | 762.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 762.15 | 763.98 | 762.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:00:00 | 762.15 | 763.98 | 762.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 763.65 | 763.92 | 762.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 761.60 | 763.92 | 762.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 762.50 | 763.63 | 762.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 762.50 | 763.63 | 762.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 763.00 | 763.51 | 762.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 767.70 | 763.51 | 762.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 768.75 | 764.56 | 763.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 772.00 | 765.95 | 764.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:30:00 | 772.00 | 767.62 | 765.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:30:00 | 770.80 | 768.89 | 766.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 14:30:00 | 773.35 | 771.29 | 768.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 779.25 | 775.18 | 771.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 780.00 | 775.18 | 771.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 775.95 | 777.98 | 774.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 775.95 | 777.98 | 774.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 773.05 | 776.99 | 774.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 773.05 | 776.99 | 774.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 771.60 | 775.91 | 773.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 771.60 | 775.91 | 773.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 771.85 | 773.41 | 773.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 768.00 | 773.41 | 773.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-22 11:15:00 | 770.00 | 772.72 | 772.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 770.00 | 772.72 | 772.88 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 775.00 | 773.00 | 772.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 783.45 | 775.09 | 773.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 779.95 | 786.00 | 782.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 779.95 | 786.00 | 782.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 779.95 | 786.00 | 782.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 779.95 | 786.00 | 782.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 775.65 | 783.93 | 782.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 780.45 | 783.15 | 781.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 781.05 | 783.15 | 781.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:30:00 | 781.55 | 782.10 | 781.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 775.00 | 780.68 | 780.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 775.00 | 780.68 | 780.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 769.50 | 777.53 | 779.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 768.15 | 749.50 | 751.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 768.15 | 749.50 | 751.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 768.15 | 749.50 | 751.62 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 772.00 | 754.00 | 753.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 779.00 | 759.00 | 755.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 762.50 | 768.00 | 762.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 762.50 | 768.00 | 762.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 762.50 | 768.00 | 762.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 749.55 | 768.00 | 762.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 726.90 | 759.78 | 759.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 726.90 | 759.78 | 759.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 716.10 | 751.04 | 755.27 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 761.95 | 751.75 | 751.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 773.00 | 758.73 | 754.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 10:15:00 | 863.75 | 864.51 | 839.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 11:00:00 | 863.75 | 864.51 | 839.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 853.50 | 855.39 | 849.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:30:00 | 865.00 | 856.20 | 850.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 848.15 | 854.97 | 850.65 | SL hit (close<static) qty=1.00 sl=849.05 alert=retest2 |

### Cycle 100 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 858.10 | 864.06 | 864.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 856.80 | 862.61 | 864.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 11:15:00 | 874.50 | 864.50 | 864.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 874.50 | 864.50 | 864.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 874.50 | 864.50 | 864.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 874.50 | 864.50 | 864.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 871.00 | 865.80 | 865.14 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 861.55 | 864.96 | 865.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 848.50 | 860.24 | 862.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 853.40 | 852.33 | 857.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 10:45:00 | 851.45 | 852.33 | 857.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 857.80 | 853.42 | 857.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 848.10 | 854.70 | 856.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 885.00 | 856.94 | 855.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 885.00 | 856.94 | 855.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 886.10 | 862.77 | 858.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 14:15:00 | 864.80 | 868.49 | 863.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 864.80 | 868.49 | 863.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 864.80 | 868.49 | 863.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 864.80 | 868.49 | 863.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 868.95 | 868.58 | 863.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 871.95 | 868.58 | 863.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 10:15:00 | 852.65 | 865.73 | 863.43 | SL hit (close<static) qty=1.00 sl=860.10 alert=retest2 |

### Cycle 104 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 850.25 | 860.95 | 861.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 843.60 | 857.48 | 859.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 847.00 | 842.48 | 848.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 847.00 | 842.48 | 848.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 847.00 | 842.48 | 848.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 846.95 | 842.48 | 848.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 842.65 | 842.51 | 847.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:30:00 | 839.50 | 841.77 | 846.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 855.25 | 844.18 | 846.95 | SL hit (close>static) qty=1.00 sl=847.60 alert=retest2 |

### Cycle 105 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 854.10 | 849.66 | 849.10 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 842.00 | 848.11 | 848.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 13:15:00 | 836.30 | 841.83 | 844.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 799.20 | 797.09 | 804.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 14:00:00 | 799.20 | 797.09 | 804.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 807.00 | 799.08 | 804.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:45:00 | 808.50 | 799.08 | 804.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 808.00 | 800.86 | 804.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 806.50 | 800.86 | 804.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 800.95 | 799.11 | 802.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:00:00 | 800.95 | 799.11 | 802.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 797.90 | 798.86 | 802.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:45:00 | 802.45 | 798.86 | 802.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 802.50 | 798.93 | 801.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 802.50 | 798.93 | 801.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 802.00 | 799.54 | 801.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 805.50 | 799.54 | 801.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 801.40 | 799.91 | 801.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 807.40 | 799.91 | 801.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 798.15 | 799.56 | 801.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:45:00 | 802.15 | 799.56 | 801.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 799.95 | 799.19 | 800.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:45:00 | 800.10 | 799.19 | 800.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 793.85 | 797.59 | 799.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:15:00 | 790.60 | 797.13 | 799.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 793.00 | 794.06 | 796.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:30:00 | 793.30 | 793.55 | 796.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 793.50 | 793.84 | 796.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 795.50 | 794.17 | 795.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 795.50 | 794.17 | 795.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 799.40 | 795.22 | 796.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 14:45:00 | 795.45 | 795.59 | 796.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 801.60 | 796.86 | 796.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 801.60 | 796.86 | 796.74 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 795.70 | 796.63 | 796.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 794.00 | 796.10 | 796.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 15:15:00 | 795.25 | 795.00 | 795.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 09:15:00 | 790.50 | 795.00 | 795.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 784.50 | 792.90 | 794.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 783.10 | 790.27 | 793.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:00:00 | 782.55 | 785.48 | 789.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 10:15:00 | 794.90 | 784.10 | 783.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 794.90 | 784.10 | 783.23 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 779.50 | 787.12 | 787.13 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 791.10 | 787.42 | 787.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 13:15:00 | 796.20 | 789.18 | 788.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 817.00 | 820.61 | 812.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 817.00 | 820.61 | 812.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 817.30 | 818.88 | 814.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 815.75 | 818.88 | 814.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 825.20 | 826.29 | 823.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 825.20 | 826.29 | 823.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 821.45 | 825.32 | 823.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 821.45 | 825.32 | 823.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 822.40 | 824.74 | 823.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 820.70 | 824.74 | 823.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 824.75 | 824.74 | 823.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:30:00 | 820.15 | 824.74 | 823.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 824.90 | 824.77 | 823.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 824.45 | 824.77 | 823.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 822.55 | 824.22 | 823.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:15:00 | 824.05 | 824.22 | 823.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 812.45 | 824.59 | 825.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 812.45 | 824.59 | 825.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 807.85 | 821.24 | 823.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 823.70 | 815.69 | 819.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 823.70 | 815.69 | 819.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 823.70 | 815.69 | 819.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 823.70 | 815.69 | 819.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 822.25 | 817.00 | 819.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:45:00 | 816.15 | 816.96 | 819.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 825.90 | 817.44 | 818.13 | SL hit (close>static) qty=1.00 sl=825.70 alert=retest2 |

### Cycle 113 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 822.75 | 819.30 | 818.90 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 810.95 | 818.28 | 818.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 806.00 | 813.32 | 816.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 11:15:00 | 801.60 | 801.20 | 804.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 11:45:00 | 802.05 | 801.20 | 804.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 803.30 | 801.71 | 804.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 803.30 | 801.71 | 804.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 802.00 | 801.77 | 804.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 801.10 | 801.77 | 804.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 799.65 | 801.48 | 803.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:45:00 | 802.20 | 801.48 | 803.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 793.00 | 787.17 | 790.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:45:00 | 791.75 | 787.17 | 790.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 792.25 | 788.19 | 790.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 792.70 | 788.19 | 790.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 803.05 | 794.60 | 793.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 817.30 | 799.14 | 795.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 812.35 | 812.65 | 804.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 812.35 | 812.65 | 804.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 809.45 | 811.43 | 806.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 808.00 | 811.43 | 806.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 811.20 | 811.62 | 809.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:45:00 | 813.00 | 810.98 | 809.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 11:15:00 | 815.15 | 818.48 | 818.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 815.15 | 818.48 | 818.77 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 822.50 | 819.47 | 819.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 823.80 | 820.34 | 819.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 11:15:00 | 820.55 | 820.58 | 819.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 12:00:00 | 820.55 | 820.58 | 819.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 817.55 | 819.98 | 819.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 817.55 | 819.98 | 819.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 822.20 | 820.42 | 819.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:45:00 | 828.40 | 822.62 | 821.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 13:45:00 | 829.95 | 825.81 | 823.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 12:15:00 | 819.95 | 822.19 | 822.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 819.95 | 822.19 | 822.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 817.00 | 821.15 | 821.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 824.00 | 821.42 | 821.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 824.00 | 821.42 | 821.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 824.00 | 821.42 | 821.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 827.55 | 821.42 | 821.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 826.05 | 822.34 | 822.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 829.05 | 824.59 | 823.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 831.90 | 832.91 | 829.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:45:00 | 830.95 | 832.91 | 829.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 833.50 | 832.88 | 829.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:45:00 | 835.10 | 833.38 | 830.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:00:00 | 835.00 | 833.55 | 831.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 836.60 | 833.99 | 832.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 834.70 | 834.84 | 832.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 840.45 | 837.42 | 834.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:45:00 | 833.75 | 837.42 | 834.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 830.90 | 841.95 | 839.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 830.90 | 841.95 | 839.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 836.00 | 840.76 | 839.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 841.20 | 840.18 | 839.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 13:15:00 | 834.90 | 838.33 | 838.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 834.90 | 838.33 | 838.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 830.10 | 836.68 | 837.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 836.40 | 834.43 | 836.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 836.40 | 834.43 | 836.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 836.40 | 834.43 | 836.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 836.40 | 834.43 | 836.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 830.50 | 833.64 | 835.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:30:00 | 828.80 | 833.11 | 835.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 843.00 | 835.66 | 835.78 | SL hit (close>static) qty=1.00 sl=836.40 alert=retest2 |

### Cycle 121 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 843.20 | 837.17 | 836.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 847.00 | 842.18 | 839.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 841.70 | 843.47 | 841.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 841.70 | 843.47 | 841.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 841.70 | 843.47 | 841.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 841.70 | 843.47 | 841.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 838.45 | 842.47 | 840.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 838.45 | 842.47 | 840.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 838.50 | 841.68 | 840.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 837.60 | 840.60 | 840.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 836.25 | 839.73 | 839.94 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 842.00 | 839.71 | 839.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 849.00 | 841.57 | 840.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 847.30 | 848.25 | 845.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:45:00 | 847.95 | 848.25 | 845.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 846.75 | 849.21 | 847.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 846.65 | 849.21 | 847.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 850.75 | 849.52 | 847.58 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 842.10 | 847.99 | 848.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 839.50 | 846.29 | 847.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 831.70 | 830.64 | 836.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 831.70 | 830.64 | 836.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 831.70 | 830.64 | 836.23 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 846.55 | 838.24 | 837.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 849.90 | 842.68 | 840.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 848.25 | 848.66 | 845.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 848.25 | 848.66 | 845.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 865.45 | 852.63 | 848.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 10:30:00 | 867.00 | 855.50 | 849.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:00:00 | 867.00 | 855.50 | 849.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 872.45 | 857.86 | 854.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:00:00 | 867.95 | 859.88 | 855.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 860.70 | 860.76 | 857.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 865.30 | 861.67 | 858.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:45:00 | 865.50 | 864.08 | 859.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:15:00 | 865.45 | 863.77 | 860.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:45:00 | 866.15 | 863.95 | 860.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 872.85 | 874.69 | 870.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 872.85 | 874.69 | 870.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 875.90 | 874.93 | 871.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 873.35 | 874.93 | 871.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 872.70 | 874.48 | 871.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 872.70 | 874.48 | 871.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 874.50 | 874.49 | 871.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 862.30 | 874.49 | 871.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 875.95 | 874.78 | 872.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 870.55 | 874.78 | 872.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 878.20 | 875.46 | 872.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 858.45 | 869.70 | 870.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 858.45 | 869.70 | 870.96 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 862.05 | 861.07 | 861.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 15:15:00 | 862.55 | 861.37 | 861.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 861.60 | 863.25 | 862.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 11:15:00 | 861.60 | 863.25 | 862.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 861.60 | 863.25 | 862.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 862.30 | 863.25 | 862.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 858.35 | 862.27 | 861.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:00:00 | 858.35 | 862.27 | 861.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 856.95 | 861.20 | 861.43 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 866.70 | 861.78 | 861.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 869.50 | 864.08 | 862.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 15:15:00 | 862.35 | 864.74 | 863.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 15:15:00 | 862.35 | 864.74 | 863.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 862.35 | 864.74 | 863.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 862.45 | 864.29 | 863.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 870.00 | 865.43 | 863.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 871.60 | 867.58 | 865.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:30:00 | 870.40 | 868.07 | 866.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 871.85 | 868.27 | 866.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:45:00 | 870.05 | 869.38 | 867.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 869.05 | 869.32 | 867.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 864.70 | 869.32 | 867.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 854.35 | 866.32 | 866.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 854.35 | 866.32 | 866.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 849.95 | 863.05 | 865.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 853.35 | 852.51 | 857.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 853.35 | 852.51 | 857.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 854.05 | 852.82 | 857.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 856.95 | 852.82 | 857.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 860.05 | 854.26 | 857.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 860.65 | 854.26 | 857.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 851.30 | 853.67 | 857.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 858.00 | 853.67 | 857.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 839.15 | 833.16 | 838.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:15:00 | 838.50 | 833.16 | 838.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 839.05 | 834.34 | 838.75 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 846.65 | 840.45 | 840.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 11:15:00 | 850.05 | 843.71 | 841.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 13:15:00 | 843.65 | 844.27 | 842.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 13:15:00 | 843.65 | 844.27 | 842.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 843.65 | 844.27 | 842.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:00:00 | 843.65 | 844.27 | 842.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 845.95 | 844.60 | 842.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:15:00 | 843.05 | 844.60 | 842.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 843.05 | 844.29 | 842.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:30:00 | 840.15 | 843.23 | 842.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 840.70 | 842.73 | 842.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 839.35 | 842.73 | 842.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 845.25 | 843.23 | 842.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:45:00 | 841.15 | 843.23 | 842.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 847.65 | 844.12 | 843.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:30:00 | 843.55 | 844.12 | 843.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 846.25 | 847.70 | 845.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:45:00 | 845.85 | 847.70 | 845.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 850.00 | 856.27 | 852.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 850.00 | 856.27 | 852.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 848.60 | 854.74 | 851.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 848.60 | 854.74 | 851.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 850.35 | 853.86 | 851.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 13:45:00 | 850.85 | 854.61 | 852.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 874.05 | 876.19 | 876.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 874.05 | 876.19 | 876.29 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 877.15 | 876.38 | 876.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 881.65 | 877.44 | 876.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 876.45 | 878.63 | 877.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 876.45 | 878.63 | 877.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 876.45 | 878.63 | 877.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 876.45 | 878.63 | 877.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 875.30 | 877.96 | 877.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 872.30 | 877.96 | 877.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 871.90 | 876.75 | 877.02 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 13:15:00 | 878.20 | 877.23 | 877.20 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 876.95 | 877.18 | 877.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 15:15:00 | 874.20 | 876.58 | 876.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 869.70 | 866.65 | 869.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 14:15:00 | 869.70 | 866.65 | 869.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 869.70 | 866.65 | 869.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:45:00 | 871.90 | 866.65 | 869.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 872.90 | 867.90 | 869.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 908.45 | 867.90 | 869.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 917.90 | 877.90 | 873.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 10:15:00 | 928.95 | 917.35 | 910.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 959.70 | 961.80 | 946.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 959.70 | 961.80 | 946.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 969.50 | 962.35 | 954.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:15:00 | 970.30 | 962.35 | 954.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:45:00 | 971.65 | 964.04 | 956.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 973.00 | 964.04 | 956.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 14:15:00 | 1023.00 | 1028.92 | 1029.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 14:15:00 | 1023.00 | 1028.92 | 1029.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 1013.40 | 1021.54 | 1025.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 1015.15 | 1014.18 | 1019.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 12:00:00 | 1015.15 | 1014.18 | 1019.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1014.35 | 1014.10 | 1017.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:30:00 | 1012.10 | 1013.44 | 1016.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 15:15:00 | 1010.40 | 1012.59 | 1014.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1025.10 | 1014.74 | 1015.50 | SL hit (close>static) qty=1.00 sl=1025.00 alert=retest2 |

### Cycle 139 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 1023.50 | 1016.49 | 1016.23 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 997.50 | 1014.74 | 1015.88 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 1026.45 | 1012.25 | 1011.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 1032.30 | 1016.26 | 1013.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 1044.50 | 1044.74 | 1035.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:00:00 | 1044.50 | 1044.74 | 1035.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1034.85 | 1041.55 | 1035.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 1034.85 | 1041.55 | 1035.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1033.90 | 1040.02 | 1035.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 1033.65 | 1040.02 | 1035.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1040.35 | 1040.09 | 1036.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 1041.80 | 1039.49 | 1036.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 10:30:00 | 1041.95 | 1038.99 | 1036.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 11:15:00 | 1030.30 | 1037.25 | 1035.78 | SL hit (close<static) qty=1.00 sl=1032.00 alert=retest2 |

### Cycle 142 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 1031.45 | 1034.21 | 1034.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1024.35 | 1031.55 | 1033.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 1033.10 | 1031.48 | 1032.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 1033.10 | 1031.48 | 1032.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1033.10 | 1031.48 | 1032.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 13:30:00 | 1028.90 | 1030.84 | 1032.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:15:00 | 977.46 | 990.77 | 998.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 973.50 | 972.62 | 979.91 | SL hit (close>ema200) qty=0.50 sl=972.62 alert=retest2 |

### Cycle 143 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 978.40 | 969.28 | 968.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 982.05 | 971.84 | 969.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 980.75 | 984.15 | 979.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 980.75 | 984.15 | 979.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 980.75 | 984.15 | 979.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 980.75 | 984.15 | 979.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 959.10 | 979.14 | 978.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 959.10 | 979.14 | 978.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 968.40 | 976.99 | 977.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 954.25 | 970.76 | 974.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 922.00 | 921.41 | 930.24 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:15:00 | 904.60 | 921.41 | 930.24 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 895.00 | 885.01 | 890.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 895.00 | 885.01 | 890.88 | SL hit (close>ema400) qty=1.00 sl=890.88 alert=retest1 |

### Cycle 145 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 897.75 | 879.68 | 877.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 903.05 | 884.35 | 880.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 895.55 | 899.77 | 892.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 895.55 | 899.77 | 892.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 887.45 | 897.31 | 892.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 887.20 | 897.31 | 892.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 890.00 | 895.84 | 892.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 879.55 | 895.84 | 892.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 876.70 | 889.70 | 889.91 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 910.80 | 892.56 | 890.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 917.35 | 897.52 | 893.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 916.25 | 919.29 | 911.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 11:00:00 | 916.25 | 919.29 | 911.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 913.35 | 918.10 | 911.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 913.20 | 918.10 | 911.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 918.10 | 917.76 | 912.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 915.45 | 917.76 | 912.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 915.15 | 921.11 | 917.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:30:00 | 912.85 | 921.11 | 917.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 913.80 | 919.65 | 916.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:45:00 | 912.85 | 919.65 | 916.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 914.20 | 918.56 | 916.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 918.85 | 917.73 | 916.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 898.95 | 914.65 | 915.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 898.95 | 914.65 | 915.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 879.10 | 907.54 | 912.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 902.60 | 902.18 | 907.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 09:30:00 | 899.75 | 902.18 | 907.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 899.60 | 891.98 | 896.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:45:00 | 899.00 | 891.98 | 896.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 900.95 | 893.77 | 896.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 900.95 | 893.77 | 896.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 912.00 | 899.03 | 898.65 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 881.95 | 895.93 | 897.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 870.45 | 883.47 | 887.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 872.55 | 870.17 | 875.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 872.55 | 870.17 | 875.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 872.55 | 870.17 | 875.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 877.50 | 870.17 | 875.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 848.15 | 845.19 | 853.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 853.30 | 845.19 | 853.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 851.10 | 846.44 | 852.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 853.95 | 846.44 | 852.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 853.85 | 847.92 | 852.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 859.75 | 847.92 | 852.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 860.65 | 850.47 | 853.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 860.65 | 850.47 | 853.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 850.75 | 847.45 | 850.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:45:00 | 850.25 | 847.45 | 850.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 847.15 | 847.39 | 850.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 855.20 | 847.39 | 850.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 865.65 | 851.04 | 851.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 865.65 | 851.04 | 851.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 867.00 | 854.24 | 852.89 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 853.50 | 858.19 | 858.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 843.05 | 848.42 | 850.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 845.70 | 845.28 | 848.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 845.70 | 845.28 | 848.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 834.70 | 843.17 | 846.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 827.25 | 843.17 | 846.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:30:00 | 827.20 | 827.62 | 834.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 13:15:00 | 834.20 | 828.77 | 833.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 13:45:00 | 833.80 | 831.10 | 833.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 843.40 | 833.56 | 834.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:15:00 | 834.55 | 833.56 | 834.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 844.45 | 835.90 | 835.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 844.45 | 835.90 | 835.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 850.90 | 838.90 | 837.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 12:15:00 | 839.25 | 839.51 | 837.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 12:15:00 | 839.25 | 839.51 | 837.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 839.25 | 839.51 | 837.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 839.25 | 839.51 | 837.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 874.90 | 875.45 | 868.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 875.45 | 875.45 | 868.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 865.45 | 872.89 | 869.40 | SL hit (close<static) qty=1.00 sl=867.45 alert=retest2 |

### Cycle 154 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 858.00 | 866.06 | 867.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 853.65 | 861.59 | 864.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 816.70 | 809.33 | 821.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:00:00 | 816.70 | 809.33 | 821.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 815.70 | 810.60 | 820.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 822.00 | 810.60 | 820.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 830.60 | 814.60 | 821.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 830.60 | 814.60 | 821.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 831.25 | 817.93 | 822.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 831.25 | 817.93 | 822.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 842.50 | 828.39 | 826.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 861.05 | 841.28 | 834.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 852.05 | 853.08 | 844.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 852.05 | 853.08 | 844.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 849.90 | 852.50 | 847.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:00:00 | 849.90 | 852.50 | 847.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 847.05 | 851.41 | 847.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:15:00 | 847.35 | 851.41 | 847.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 849.50 | 851.03 | 847.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 13:15:00 | 855.20 | 850.55 | 848.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 859.20 | 852.28 | 849.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 850.50 | 859.66 | 860.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 850.50 | 859.66 | 860.54 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 874.35 | 863.05 | 861.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 12:15:00 | 879.30 | 866.30 | 863.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 888.30 | 892.44 | 884.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 11:00:00 | 888.30 | 892.44 | 884.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 936.10 | 917.91 | 906.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 954.95 | 936.55 | 922.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 12:15:00 | 917.00 | 925.14 | 925.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 917.00 | 925.14 | 925.65 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 934.95 | 925.29 | 925.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 941.40 | 930.71 | 927.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-08 14:15:00 | 933.65 | 933.74 | 930.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-08 15:00:00 | 933.65 | 933.74 | 930.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 931.40 | 933.44 | 930.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 931.40 | 933.44 | 930.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 931.35 | 933.02 | 930.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 931.35 | 933.02 | 930.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 941.25 | 934.67 | 931.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 14:00:00 | 948.00 | 938.32 | 933.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:30:00 | 947.00 | 941.55 | 938.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 957.95 | 978.04 | 979.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 957.95 | 978.04 | 979.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 952.20 | 962.49 | 967.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 944.05 | 943.23 | 949.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 15:00:00 | 944.05 | 943.23 | 949.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 952.70 | 944.88 | 948.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 952.70 | 944.88 | 948.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 948.55 | 945.62 | 948.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:15:00 | 945.90 | 945.88 | 948.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 14:15:00 | 945.20 | 946.76 | 948.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 945.00 | 946.83 | 948.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 945.55 | 948.04 | 948.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 943.65 | 947.16 | 948.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:30:00 | 941.35 | 946.12 | 947.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:30:00 | 941.85 | 945.59 | 947.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 948.20 | 947.93 | 947.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 948.20 | 947.93 | 947.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 963.95 | 951.13 | 949.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 954.25 | 956.84 | 953.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 954.25 | 956.84 | 953.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 954.25 | 956.84 | 953.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 952.00 | 956.84 | 953.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 946.00 | 954.67 | 952.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 933.85 | 954.67 | 952.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 935.20 | 950.78 | 951.13 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 950.00 | 947.12 | 947.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 965.00 | 951.15 | 948.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 1005.80 | 1005.97 | 997.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 14:45:00 | 1005.65 | 1005.97 | 997.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1009.00 | 1006.00 | 999.11 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 989.10 | 997.21 | 998.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 982.55 | 994.27 | 996.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 984.75 | 981.99 | 987.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 984.75 | 981.99 | 987.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 984.75 | 981.99 | 987.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 984.85 | 981.99 | 987.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 990.10 | 983.61 | 987.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 990.10 | 983.61 | 987.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 989.90 | 984.87 | 987.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 983.50 | 984.65 | 987.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 979.05 | 984.94 | 986.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 998.50 | 987.66 | 987.71 | SL hit (close>static) qty=1.00 sl=992.40 alert=retest2 |

### Cycle 165 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 996.70 | 989.46 | 988.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1006.00 | 992.77 | 990.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 986.20 | 996.89 | 993.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 986.20 | 996.89 | 993.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 986.20 | 996.89 | 993.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 986.20 | 996.89 | 993.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 980.25 | 993.56 | 992.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 980.25 | 993.56 | 992.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 982.50 | 991.35 | 991.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 972.80 | 983.43 | 987.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 969.00 | 964.46 | 970.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 969.00 | 964.46 | 970.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 969.00 | 964.46 | 970.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 971.45 | 964.46 | 970.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 968.65 | 965.30 | 970.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 966.10 | 965.30 | 970.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 962.05 | 964.65 | 969.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 959.00 | 961.74 | 967.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 975.50 | 960.69 | 962.38 | SL hit (close>static) qty=1.00 sl=969.95 alert=retest2 |

### Cycle 167 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 973.10 | 964.66 | 963.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 983.75 | 968.48 | 965.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 988.40 | 988.51 | 981.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:45:00 | 987.75 | 988.51 | 981.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 982.00 | 987.21 | 981.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 980.50 | 987.21 | 981.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 991.20 | 988.01 | 982.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 978.15 | 988.01 | 982.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 996.90 | 1001.88 | 997.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 996.90 | 1001.88 | 997.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1000.00 | 1001.51 | 997.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 996.00 | 1001.51 | 997.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1059.80 | 1066.17 | 1056.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 1062.55 | 1066.17 | 1056.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1058.65 | 1064.67 | 1056.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1058.65 | 1064.67 | 1056.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1052.95 | 1062.33 | 1056.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1052.95 | 1062.33 | 1056.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1054.10 | 1060.68 | 1056.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1051.00 | 1060.68 | 1056.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1051.00 | 1058.74 | 1055.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1047.45 | 1058.74 | 1055.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1049.45 | 1056.89 | 1055.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 1057.35 | 1057.31 | 1055.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 1048.50 | 1063.46 | 1063.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 1048.50 | 1063.46 | 1063.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1039.05 | 1053.72 | 1058.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1025.90 | 1024.65 | 1034.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 1025.90 | 1024.65 | 1034.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1008.75 | 1021.72 | 1030.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:30:00 | 1026.80 | 1021.72 | 1030.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1016.25 | 1016.54 | 1023.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 1011.05 | 1016.05 | 1023.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 1010.10 | 1015.74 | 1022.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 09:45:00 | 1007.65 | 1013.82 | 1020.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 1009.60 | 1015.52 | 1019.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1016.30 | 1015.68 | 1018.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1016.30 | 1015.68 | 1018.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1021.10 | 1016.76 | 1019.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 1014.10 | 1018.13 | 1019.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 1041.75 | 1030.77 | 1026.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 1042.10 | 1045.81 | 1039.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:00:00 | 1042.10 | 1045.81 | 1039.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1077.70 | 1083.36 | 1078.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 1077.50 | 1083.36 | 1078.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1072.20 | 1081.13 | 1077.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 1072.20 | 1081.13 | 1077.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1072.90 | 1079.48 | 1077.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 1072.90 | 1079.48 | 1077.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1072.90 | 1077.45 | 1076.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1077.90 | 1077.45 | 1076.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:00:00 | 1083.00 | 1078.56 | 1077.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 1069.10 | 1082.51 | 1081.73 | SL hit (close<static) qty=1.00 sl=1070.70 alert=retest2 |

### Cycle 170 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 1074.10 | 1080.37 | 1081.15 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1086.00 | 1081.92 | 1081.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1096.70 | 1084.87 | 1083.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 1099.60 | 1099.91 | 1093.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 1099.60 | 1099.91 | 1093.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1099.60 | 1099.91 | 1093.76 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 1086.00 | 1091.64 | 1091.68 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1099.00 | 1093.11 | 1092.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 1105.50 | 1095.59 | 1093.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 14:15:00 | 1134.90 | 1139.22 | 1128.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1134.90 | 1139.22 | 1128.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1123.00 | 1135.97 | 1127.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 1137.20 | 1136.64 | 1128.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 1135.50 | 1135.81 | 1130.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 1151.90 | 1155.93 | 1156.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1151.90 | 1155.93 | 1156.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 1149.80 | 1154.71 | 1155.71 | Break + close below crossover candle low |

### Cycle 175 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1172.10 | 1156.36 | 1156.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1190.80 | 1166.23 | 1160.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1170.80 | 1177.57 | 1169.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1170.80 | 1177.57 | 1169.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1170.80 | 1177.57 | 1169.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1170.80 | 1177.57 | 1169.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1168.60 | 1175.78 | 1169.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1165.10 | 1175.78 | 1169.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1178.50 | 1176.32 | 1170.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 1183.10 | 1178.24 | 1171.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 1182.70 | 1181.81 | 1176.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 1181.20 | 1181.69 | 1177.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 1181.50 | 1181.17 | 1177.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1179.00 | 1180.73 | 1177.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1176.20 | 1180.73 | 1177.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1174.20 | 1179.43 | 1177.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1167.40 | 1179.43 | 1177.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1170.00 | 1177.54 | 1176.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 1158.50 | 1171.87 | 1174.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1167.60 | 1160.45 | 1166.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 1167.60 | 1160.45 | 1166.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1167.60 | 1160.45 | 1166.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 1163.80 | 1160.45 | 1166.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1172.80 | 1162.92 | 1167.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 1172.80 | 1162.92 | 1167.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1169.90 | 1166.25 | 1167.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:30:00 | 1175.60 | 1166.25 | 1167.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1168.40 | 1167.81 | 1168.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 1168.40 | 1167.81 | 1168.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 1181.50 | 1170.55 | 1169.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 1183.20 | 1173.08 | 1170.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 1182.40 | 1184.37 | 1178.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 1182.40 | 1184.37 | 1178.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1182.40 | 1184.37 | 1178.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 1182.40 | 1184.37 | 1178.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1186.00 | 1184.15 | 1179.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 1181.20 | 1184.15 | 1179.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1183.50 | 1191.04 | 1185.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 1183.50 | 1191.04 | 1185.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1183.00 | 1189.43 | 1185.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 1185.50 | 1189.43 | 1185.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1187.60 | 1189.07 | 1185.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1191.10 | 1189.51 | 1186.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 15:15:00 | 1170.10 | 1184.04 | 1184.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1170.10 | 1184.04 | 1184.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1163.90 | 1175.81 | 1179.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1171.00 | 1164.72 | 1172.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1171.00 | 1164.72 | 1172.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1171.00 | 1164.72 | 1172.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:15:00 | 1144.80 | 1152.86 | 1157.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:45:00 | 1144.00 | 1148.66 | 1154.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 1087.56 | 1129.59 | 1143.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 1086.80 | 1129.59 | 1143.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1070.00 | 1068.47 | 1089.68 | SL hit (close>ema200) qty=0.50 sl=1068.47 alert=retest2 |

### Cycle 179 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 1082.60 | 1073.93 | 1072.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1100.00 | 1085.43 | 1080.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 1103.00 | 1104.44 | 1097.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 1082.00 | 1104.44 | 1097.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1085.50 | 1100.65 | 1096.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 1081.50 | 1100.65 | 1096.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1082.10 | 1096.94 | 1095.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1082.20 | 1096.94 | 1095.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 1071.60 | 1091.87 | 1093.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1063.30 | 1075.07 | 1082.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 1039.90 | 1038.96 | 1049.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:15:00 | 1031.10 | 1038.96 | 1049.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1045.70 | 1040.70 | 1048.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 1045.50 | 1040.70 | 1048.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1040.70 | 1041.24 | 1046.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 1048.00 | 1041.24 | 1046.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 1046.60 | 1042.31 | 1046.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 1046.60 | 1042.31 | 1046.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1044.60 | 1042.77 | 1046.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1048.40 | 1042.77 | 1046.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1059.40 | 1046.10 | 1047.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1059.40 | 1046.10 | 1047.71 | SL hit (close>ema400) qty=1.00 sl=1047.71 alert=retest1 |

### Cycle 181 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1063.10 | 1050.94 | 1049.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1069.00 | 1058.74 | 1054.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1069.90 | 1072.67 | 1065.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1069.90 | 1072.67 | 1065.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1074.50 | 1075.32 | 1070.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1080.00 | 1075.32 | 1070.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1079.20 | 1080.64 | 1076.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1067.00 | 1077.91 | 1075.25 | SL hit (close<static) qty=1.00 sl=1069.20 alert=retest2 |

### Cycle 182 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 1068.50 | 1073.24 | 1073.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 15:15:00 | 1066.00 | 1071.79 | 1072.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1076.70 | 1072.77 | 1073.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1076.70 | 1072.77 | 1073.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1076.70 | 1072.77 | 1073.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1076.70 | 1072.77 | 1073.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1068.20 | 1071.86 | 1072.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 1063.90 | 1071.86 | 1072.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 1053.80 | 1045.42 | 1044.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 1053.80 | 1045.42 | 1044.28 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1037.50 | 1043.92 | 1044.71 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1082.50 | 1051.49 | 1048.00 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1052.20 | 1056.76 | 1056.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 1049.40 | 1054.81 | 1055.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 1054.70 | 1052.24 | 1054.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1054.70 | 1052.24 | 1054.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1054.70 | 1052.24 | 1054.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 1056.00 | 1052.24 | 1054.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1060.80 | 1053.96 | 1054.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1060.80 | 1053.96 | 1054.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1058.90 | 1054.94 | 1055.25 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 1060.20 | 1056.00 | 1055.70 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 1052.30 | 1055.26 | 1055.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1045.00 | 1053.21 | 1054.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1050.70 | 1050.52 | 1052.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 1050.70 | 1050.52 | 1052.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1022.40 | 1015.78 | 1024.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 1022.80 | 1015.78 | 1024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 991.80 | 986.21 | 994.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 996.35 | 986.21 | 994.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 997.15 | 988.40 | 994.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 988.20 | 986.72 | 992.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:45:00 | 984.00 | 985.45 | 989.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1001.80 | 989.90 | 988.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 1001.80 | 989.90 | 988.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 1011.80 | 994.28 | 991.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1000.05 | 1001.71 | 997.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1000.05 | 1001.71 | 997.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1000.05 | 1001.71 | 997.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 1000.45 | 1001.71 | 997.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 999.50 | 1000.92 | 997.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 998.25 | 1000.92 | 997.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 999.40 | 1000.62 | 997.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 999.40 | 1000.62 | 997.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 998.10 | 1000.12 | 997.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 998.10 | 1000.12 | 997.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1000.00 | 1000.09 | 997.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1001.55 | 1000.09 | 997.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1000.95 | 1000.26 | 998.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 1006.30 | 1001.25 | 998.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:45:00 | 1006.40 | 1003.29 | 1000.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 1009.10 | 1003.29 | 1000.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 1008.30 | 1005.07 | 1001.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1023.25 | 1009.49 | 1004.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1016.00 | 1007.38 | 1007.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 1030.40 | 1013.65 | 1010.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1020.45 | 1024.72 | 1019.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1020.45 | 1024.72 | 1019.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1020.45 | 1024.72 | 1019.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 1027.25 | 1023.12 | 1020.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 1011.85 | 1019.41 | 1019.22 | SL hit (close<static) qty=1.00 sl=1012.55 alert=retest2 |

### Cycle 192 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1058.75 | 1052.60 | 1051.82 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1046.40 | 1050.75 | 1051.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1046.00 | 1049.80 | 1050.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 1036.30 | 1032.81 | 1038.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1036.30 | 1032.81 | 1038.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1036.30 | 1032.81 | 1038.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1036.30 | 1032.81 | 1038.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1034.00 | 1033.05 | 1038.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 1033.60 | 1033.05 | 1038.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1017.60 | 1026.79 | 1032.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 1014.00 | 1026.79 | 1032.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 1032.70 | 1030.73 | 1030.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 1032.70 | 1030.73 | 1030.63 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 1026.80 | 1029.94 | 1030.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 1020.30 | 1028.01 | 1029.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 14:15:00 | 1006.80 | 1005.78 | 1012.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 15:00:00 | 1006.80 | 1005.78 | 1012.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 997.70 | 985.54 | 991.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 997.70 | 985.54 | 991.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 996.00 | 987.63 | 992.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 997.10 | 987.63 | 992.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 995.70 | 991.30 | 993.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:45:00 | 994.70 | 991.30 | 993.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 994.70 | 991.98 | 993.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:15:00 | 994.90 | 991.98 | 993.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 985.50 | 991.15 | 992.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 983.90 | 988.72 | 991.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:15:00 | 984.40 | 987.99 | 990.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:15:00 | 984.10 | 987.44 | 990.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:45:00 | 984.10 | 986.61 | 989.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 984.10 | 985.53 | 988.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1000.00 | 991.54 | 989.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1011.90 | 1013.19 | 1006.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:45:00 | 1008.90 | 1013.19 | 1006.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1007.60 | 1011.72 | 1007.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 1000.20 | 1011.72 | 1007.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1001.00 | 1009.58 | 1006.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 12:45:00 | 1017.30 | 1010.91 | 1008.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 1017.10 | 1011.63 | 1009.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 1017.10 | 1013.75 | 1011.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 1016.20 | 1013.86 | 1011.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1018.00 | 1016.11 | 1013.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1029.60 | 1016.11 | 1013.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1025.20 | 1020.65 | 1016.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1027.20 | 1025.98 | 1023.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:45:00 | 1023.10 | 1024.77 | 1023.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1019.00 | 1023.61 | 1022.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:30:00 | 1019.20 | 1023.61 | 1022.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1015.50 | 1021.03 | 1021.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1021.10 | 1021.04 | 1021.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1021.10 | 1021.04 | 1021.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1021.10 | 1021.04 | 1021.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1021.10 | 1021.04 | 1021.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1022.00 | 1021.23 | 1021.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1015.00 | 1021.23 | 1021.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1025.40 | 1020.61 | 1020.90 | SL hit (close>static) qty=1.00 sl=1024.50 alert=retest2 |

### Cycle 199 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 1025.00 | 1021.49 | 1021.27 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 1011.80 | 1019.55 | 1020.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 1009.90 | 1014.07 | 1016.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 1014.10 | 1012.61 | 1015.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 1014.10 | 1012.61 | 1015.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1014.10 | 1012.61 | 1015.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1014.10 | 1012.61 | 1015.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1006.10 | 1009.97 | 1013.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:15:00 | 1005.10 | 1009.97 | 1013.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 1005.50 | 1009.08 | 1012.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1023.70 | 1006.98 | 1006.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 1023.70 | 1006.98 | 1006.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1052.50 | 1029.92 | 1021.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1058.60 | 1059.28 | 1049.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1046.90 | 1059.28 | 1049.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1041.90 | 1055.80 | 1048.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1041.90 | 1055.80 | 1048.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1045.30 | 1053.70 | 1048.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1045.30 | 1053.70 | 1048.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1050.00 | 1052.10 | 1048.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 1053.10 | 1052.74 | 1048.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 1051.90 | 1052.41 | 1049.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:15:00 | 1052.00 | 1052.41 | 1049.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1040.60 | 1049.98 | 1048.58 | SL hit (close<static) qty=1.00 sl=1046.60 alert=retest2 |

### Cycle 202 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 1046.10 | 1048.02 | 1048.08 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1053.80 | 1049.06 | 1048.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 14:15:00 | 1063.20 | 1052.28 | 1050.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 11:15:00 | 1051.40 | 1054.23 | 1052.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 1051.40 | 1054.23 | 1052.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1051.40 | 1054.23 | 1052.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 1051.40 | 1054.23 | 1052.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1050.80 | 1053.54 | 1051.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1050.30 | 1053.54 | 1051.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1051.00 | 1053.03 | 1051.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 1053.30 | 1053.03 | 1051.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1050.00 | 1052.43 | 1051.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1050.00 | 1052.43 | 1051.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1050.50 | 1052.04 | 1051.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1058.30 | 1052.04 | 1051.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 1046.00 | 1056.55 | 1055.66 | SL hit (close<static) qty=1.00 sl=1049.70 alert=retest2 |

### Cycle 204 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 1046.80 | 1054.60 | 1054.85 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 1062.60 | 1054.22 | 1053.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1074.70 | 1061.44 | 1057.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 1065.30 | 1066.41 | 1061.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 12:45:00 | 1064.60 | 1066.41 | 1061.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1064.60 | 1066.14 | 1061.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 1062.50 | 1066.14 | 1061.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1062.70 | 1064.81 | 1061.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1061.90 | 1064.81 | 1061.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1057.80 | 1063.41 | 1061.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 1057.80 | 1063.41 | 1061.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1065.00 | 1063.73 | 1061.92 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 1056.60 | 1060.86 | 1061.06 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 1063.50 | 1061.39 | 1061.28 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1052.00 | 1059.51 | 1060.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1049.30 | 1057.47 | 1059.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1056.90 | 1053.87 | 1056.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 1056.90 | 1053.87 | 1056.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1056.90 | 1053.87 | 1056.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1056.90 | 1053.87 | 1056.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1055.00 | 1054.10 | 1056.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 1058.10 | 1054.10 | 1056.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1059.90 | 1055.26 | 1056.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1059.90 | 1055.26 | 1056.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1055.90 | 1055.39 | 1056.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:30:00 | 1053.90 | 1055.03 | 1056.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 1053.60 | 1055.03 | 1056.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1048.80 | 1055.02 | 1056.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 1059.00 | 1057.02 | 1056.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1059.00 | 1057.02 | 1056.78 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1053.00 | 1056.52 | 1056.77 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 1060.00 | 1057.30 | 1057.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1088.00 | 1063.44 | 1059.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1075.60 | 1076.38 | 1069.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:15:00 | 1075.50 | 1076.38 | 1069.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1076.90 | 1077.49 | 1073.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 1087.00 | 1076.53 | 1074.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1084.10 | 1080.54 | 1077.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1085.50 | 1081.33 | 1078.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 10:00:00 | 1082.80 | 1087.03 | 1083.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 1080.10 | 1085.64 | 1083.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:45:00 | 1081.50 | 1085.64 | 1083.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1078.00 | 1084.11 | 1082.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:45:00 | 1074.90 | 1084.11 | 1082.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1088.50 | 1086.54 | 1084.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 1088.50 | 1086.54 | 1084.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1082.90 | 1085.81 | 1084.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 1082.90 | 1085.81 | 1084.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1085.00 | 1085.65 | 1084.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1080.50 | 1085.65 | 1084.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 1070.50 | 1077.92 | 1079.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 10:15:00 | 1071.50 | 1071.37 | 1075.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 1071.50 | 1071.37 | 1075.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1075.00 | 1069.79 | 1072.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1060.00 | 1069.79 | 1072.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 1065.00 | 1064.66 | 1068.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:45:00 | 1066.20 | 1065.39 | 1067.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 1067.00 | 1065.59 | 1067.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1070.70 | 1066.61 | 1068.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:30:00 | 1073.30 | 1066.61 | 1068.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1067.00 | 1066.69 | 1067.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 1056.40 | 1066.95 | 1067.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 12:15:00 | 1081.40 | 1072.78 | 1070.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 09:15:00 | 1074.00 | 1075.90 | 1072.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 1074.00 | 1075.90 | 1072.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1071.20 | 1074.96 | 1072.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1071.20 | 1074.96 | 1072.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1074.60 | 1074.89 | 1072.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 1077.90 | 1075.19 | 1073.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 09:45:00 | 1079.80 | 1078.42 | 1075.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 1075.00 | 1077.20 | 1075.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1075.00 | 1076.70 | 1075.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | SL hit (close<static) qty=1.00 sl=1070.10 alert=retest2 |

### Cycle 214 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1056.60 | 1071.61 | 1073.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 1046.00 | 1066.49 | 1070.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1073.20 | 1062.06 | 1065.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1073.20 | 1062.06 | 1065.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1073.20 | 1062.06 | 1065.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1073.20 | 1062.06 | 1065.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1055.00 | 1060.65 | 1064.69 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 11:15:00 | 1078.10 | 1067.92 | 1066.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 1083.00 | 1070.93 | 1068.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1093.70 | 1105.82 | 1094.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1093.70 | 1105.82 | 1094.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1093.70 | 1105.82 | 1094.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 1093.70 | 1105.82 | 1094.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1095.20 | 1103.69 | 1094.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1092.40 | 1103.69 | 1094.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1101.60 | 1103.28 | 1095.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 1107.70 | 1097.78 | 1095.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:00:00 | 1112.00 | 1103.55 | 1098.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1150.90 | 1165.18 | 1166.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1150.90 | 1165.18 | 1166.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1127.00 | 1149.00 | 1156.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 1146.20 | 1144.12 | 1151.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 1146.20 | 1144.12 | 1151.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1149.10 | 1144.89 | 1150.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1131.10 | 1144.89 | 1150.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 1141.60 | 1142.05 | 1146.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1131.20 | 1143.50 | 1146.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:45:00 | 1141.60 | 1140.76 | 1143.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1147.30 | 1142.07 | 1144.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1147.30 | 1142.07 | 1144.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 1137.90 | 1144.10 | 1144.78 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1159.30 | 1145.75 | 1145.07 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1143.50 | 1147.64 | 1148.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1133.80 | 1143.49 | 1145.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1131.70 | 1120.54 | 1127.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1131.70 | 1120.54 | 1127.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1131.70 | 1120.54 | 1127.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1131.70 | 1120.54 | 1127.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1123.80 | 1121.19 | 1127.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1118.50 | 1121.19 | 1127.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 1141.50 | 1126.92 | 1128.74 | SL hit (close>static) qty=1.00 sl=1131.70 alert=retest2 |

### Cycle 221 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1138.70 | 1131.10 | 1130.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 1149.90 | 1134.86 | 1132.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 1131.00 | 1134.09 | 1132.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 1131.00 | 1134.09 | 1132.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1131.00 | 1134.09 | 1132.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1129.00 | 1134.09 | 1132.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1134.90 | 1134.25 | 1132.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 1132.40 | 1134.25 | 1132.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1133.00 | 1134.00 | 1132.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 1133.00 | 1134.00 | 1132.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 1133.90 | 1133.98 | 1132.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:30:00 | 1133.30 | 1133.98 | 1132.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1151.40 | 1150.05 | 1143.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 1143.50 | 1150.05 | 1143.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1147.10 | 1149.46 | 1144.27 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1122.90 | 1139.63 | 1140.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1118.40 | 1131.94 | 1136.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 1132.10 | 1131.90 | 1135.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 13:30:00 | 1133.60 | 1131.90 | 1135.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1097.40 | 1069.37 | 1081.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1097.40 | 1069.37 | 1081.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1106.90 | 1076.87 | 1083.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1092.90 | 1076.87 | 1083.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1096.80 | 1081.96 | 1085.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1085.00 | 1081.81 | 1084.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 1102.50 | 1089.51 | 1088.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 1102.50 | 1089.51 | 1088.03 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1054.80 | 1082.35 | 1085.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 1040.00 | 1073.88 | 1081.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 999.00 | 997.06 | 1011.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 999.00 | 997.06 | 1011.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 990.00 | 991.56 | 1003.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 975.60 | 987.31 | 1000.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 15:00:00 | 972.10 | 982.24 | 992.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 14:15:00 | 985.00 | 974.71 | 973.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 985.00 | 974.71 | 973.77 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 955.00 | 971.74 | 972.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 950.70 | 963.36 | 968.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 939.90 | 938.91 | 950.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:15:00 | 918.30 | 938.91 | 950.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 14:15:00 | 872.38 | 897.78 | 921.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 892.80 | 886.92 | 906.18 | SL hit (close>ema200) qty=0.50 sl=886.92 alert=retest1 |

### Cycle 227 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 921.30 | 909.44 | 909.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 935.00 | 920.14 | 914.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 900.30 | 916.17 | 913.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 900.30 | 916.17 | 913.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 900.30 | 916.17 | 913.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 900.30 | 916.17 | 913.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 906.40 | 914.22 | 912.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 901.20 | 914.22 | 912.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 929.00 | 917.06 | 914.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 933.50 | 917.06 | 914.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 909.40 | 918.27 | 915.55 | SL hit (close<static) qty=1.00 sl=912.60 alert=retest2 |

### Cycle 228 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 898.40 | 912.25 | 913.16 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 946.20 | 916.55 | 913.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 950.70 | 932.34 | 922.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 918.30 | 933.56 | 926.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 918.30 | 933.56 | 926.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 918.30 | 933.56 | 926.80 | EMA400 retest candle locked (from upside) |

### Cycle 230 — SELL (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 13:15:00 | 921.85 | 922.80 | 922.88 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 928.30 | 923.90 | 923.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 936.95 | 928.20 | 925.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 928.35 | 930.94 | 927.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 928.35 | 930.94 | 927.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 928.35 | 930.94 | 927.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 927.65 | 930.94 | 927.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 930.60 | 930.87 | 928.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:45:00 | 931.40 | 930.87 | 928.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 979.15 | 989.30 | 981.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 994.85 | 986.91 | 982.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 999.70 | 986.56 | 983.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 996.15 | 990.52 | 986.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:00:00 | 993.90 | 991.19 | 986.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1014.70 | 1007.71 | 1002.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 1009.60 | 1007.71 | 1002.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1006.00 | 1011.26 | 1007.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 1014.20 | 1011.85 | 1007.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 998.00 | 1009.27 | 1008.46 | SL hit (close<static) qty=1.00 sl=1003.10 alert=retest2 |

### Cycle 232 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 994.50 | 1006.32 | 1007.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 992.30 | 1001.33 | 1004.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 964.15 | 963.60 | 973.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:45:00 | 967.05 | 963.60 | 973.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 968.55 | 962.24 | 968.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 968.55 | 962.24 | 968.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 966.20 | 963.03 | 967.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:15:00 | 965.00 | 963.03 | 967.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 965.00 | 963.43 | 967.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 974.95 | 963.43 | 967.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 972.20 | 965.18 | 968.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 958.45 | 965.72 | 968.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 957.75 | 961.96 | 965.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 958.80 | 961.90 | 964.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:30:00 | 962.80 | 960.29 | 963.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 949.00 | 940.31 | 948.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 947.00 | 940.31 | 948.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 929.00 | 938.05 | 946.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 928.50 | 936.09 | 943.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:45:00 | 928.25 | 931.77 | 939.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 910.86 | 924.72 | 934.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 914.66 | 924.72 | 934.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 910.53 | 921.73 | 931.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 909.86 | 921.73 | 931.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 922.05 | 921.80 | 931.01 | SL hit (close>ema200) qty=0.50 sl=921.80 alert=retest2 |

### Cycle 233 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 943.30 | 931.17 | 930.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 951.15 | 940.12 | 935.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 953.00 | 954.94 | 948.22 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-22 12:30:00 | 799.00 | 2024-04-23 13:15:00 | 812.05 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-05-08 13:30:00 | 757.85 | 2024-05-13 15:15:00 | 763.95 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-05-09 09:30:00 | 761.90 | 2024-05-13 15:15:00 | 763.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-05-16 15:15:00 | 772.00 | 2024-05-22 11:15:00 | 770.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-05-17 09:30:00 | 772.00 | 2024-05-22 11:15:00 | 770.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-05-17 10:30:00 | 770.80 | 2024-05-22 11:15:00 | 770.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-05-17 14:30:00 | 773.35 | 2024-05-22 11:15:00 | 770.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-05-27 09:30:00 | 780.45 | 2024-05-27 12:15:00 | 775.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-05-27 10:15:00 | 781.05 | 2024-05-27 12:15:00 | 775.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-05-27 11:30:00 | 781.55 | 2024-05-27 12:15:00 | 775.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-06-12 14:30:00 | 865.00 | 2024-06-13 09:15:00 | 848.15 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-06-13 12:00:00 | 858.55 | 2024-06-19 14:15:00 | 858.10 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-06-25 11:15:00 | 848.10 | 2024-06-26 09:15:00 | 885.00 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2024-06-27 09:15:00 | 871.95 | 2024-06-27 10:15:00 | 852.65 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-07-01 11:30:00 | 839.50 | 2024-07-01 13:15:00 | 855.25 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-07-12 11:15:00 | 790.60 | 2024-07-16 09:15:00 | 801.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-07-12 15:00:00 | 793.00 | 2024-07-16 09:15:00 | 801.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-15 09:30:00 | 793.30 | 2024-07-16 09:15:00 | 801.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-15 11:15:00 | 793.50 | 2024-07-16 09:15:00 | 801.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-07-15 14:45:00 | 795.45 | 2024-07-16 09:15:00 | 801.60 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-18 10:30:00 | 783.10 | 2024-07-23 10:15:00 | 794.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-07-18 15:00:00 | 782.55 | 2024-07-23 10:15:00 | 794.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-08-02 10:15:00 | 824.05 | 2024-08-05 10:15:00 | 812.45 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-08-06 11:45:00 | 816.15 | 2024-08-07 10:15:00 | 825.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-21 14:45:00 | 813.00 | 2024-08-26 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-08-28 09:45:00 | 828.40 | 2024-08-29 12:15:00 | 819.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-08-28 13:45:00 | 829.95 | 2024-08-29 12:15:00 | 819.95 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-03 09:45:00 | 835.10 | 2024-09-06 13:15:00 | 834.90 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-09-03 12:00:00 | 835.00 | 2024-09-06 13:15:00 | 834.90 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-09-04 09:15:00 | 836.60 | 2024-09-06 13:15:00 | 834.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-09-04 10:45:00 | 834.70 | 2024-09-06 13:15:00 | 834.90 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-09-06 12:15:00 | 841.20 | 2024-09-06 13:15:00 | 834.90 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-09-09 12:30:00 | 828.80 | 2024-09-10 09:15:00 | 843.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-09-25 10:30:00 | 867.00 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-25 11:00:00 | 867.00 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-27 09:30:00 | 872.45 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-27 11:00:00 | 867.95 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-27 15:00:00 | 865.30 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-09-30 09:45:00 | 865.50 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-30 11:15:00 | 865.45 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-30 12:45:00 | 866.15 | 2024-10-07 09:15:00 | 858.45 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-10-16 09:30:00 | 871.60 | 2024-10-17 11:15:00 | 854.35 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-16 10:30:00 | 870.40 | 2024-10-17 11:15:00 | 854.35 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-10-16 14:15:00 | 871.85 | 2024-10-17 11:15:00 | 854.35 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-10-17 09:45:00 | 870.05 | 2024-10-17 11:15:00 | 854.35 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-10-29 13:45:00 | 850.85 | 2024-11-06 10:15:00 | 874.05 | STOP_HIT | 1.00 | 2.73% |
| BUY | retest2 | 2024-11-27 10:15:00 | 970.30 | 2024-12-05 14:15:00 | 1023.00 | STOP_HIT | 1.00 | 5.43% |
| BUY | retest2 | 2024-11-27 11:45:00 | 971.65 | 2024-12-05 14:15:00 | 1023.00 | STOP_HIT | 1.00 | 5.28% |
| BUY | retest2 | 2024-11-27 12:15:00 | 973.00 | 2024-12-05 14:15:00 | 1023.00 | STOP_HIT | 1.00 | 5.14% |
| SELL | retest2 | 2024-12-10 10:30:00 | 1012.10 | 2024-12-11 09:15:00 | 1025.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-12-10 15:15:00 | 1010.40 | 2024-12-11 09:15:00 | 1025.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-18 09:30:00 | 1041.80 | 2024-12-18 11:15:00 | 1030.30 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-18 10:30:00 | 1041.95 | 2024-12-18 11:15:00 | 1030.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-19 13:30:00 | 1028.90 | 2024-12-26 10:15:00 | 977.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 13:30:00 | 1028.90 | 2024-12-30 09:15:00 | 973.50 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest1 | 2025-01-13 09:15:00 | 904.60 | 2025-01-16 09:15:00 | 895.00 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2025-01-16 13:00:00 | 890.90 | 2025-01-23 11:15:00 | 897.75 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-01-16 14:15:00 | 889.50 | 2025-01-23 11:15:00 | 897.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-02-01 09:15:00 | 918.85 | 2025-02-01 11:15:00 | 898.95 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-02-28 09:15:00 | 827.25 | 2025-03-04 09:15:00 | 844.45 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-03-03 09:30:00 | 827.20 | 2025-03-04 09:15:00 | 844.45 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-03-03 13:15:00 | 834.20 | 2025-03-04 09:15:00 | 844.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-03 13:45:00 | 833.80 | 2025-03-04 09:15:00 | 844.45 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-03-03 15:15:00 | 834.55 | 2025-03-04 09:15:00 | 844.45 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-10 09:15:00 | 875.45 | 2025-03-10 11:15:00 | 865.45 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-21 13:15:00 | 855.20 | 2025-03-26 14:15:00 | 850.50 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-03-21 14:00:00 | 859.20 | 2025-03-26 14:15:00 | 850.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-04-04 09:30:00 | 954.95 | 2025-04-07 12:15:00 | 917.00 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2025-04-09 14:00:00 | 948.00 | 2025-04-25 10:15:00 | 957.95 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-04-15 09:30:00 | 947.00 | 2025-04-25 10:15:00 | 957.95 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-05-05 12:15:00 | 945.90 | 2025-05-07 15:15:00 | 948.20 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-05-05 14:15:00 | 945.20 | 2025-05-07 15:15:00 | 948.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-05-06 09:15:00 | 945.00 | 2025-05-07 15:15:00 | 948.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-05-06 11:15:00 | 945.55 | 2025-05-07 15:15:00 | 948.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-05-06 12:30:00 | 941.35 | 2025-05-07 15:15:00 | 948.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-06 13:30:00 | 941.85 | 2025-05-07 15:15:00 | 948.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-22 12:30:00 | 983.50 | 2025-05-23 09:15:00 | 998.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-23 09:15:00 | 979.05 | 2025-05-23 09:15:00 | 998.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-05-29 12:30:00 | 959.00 | 2025-05-30 14:15:00 | 975.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-06-13 10:30:00 | 1057.35 | 2025-06-18 09:15:00 | 1048.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-23 13:30:00 | 1011.05 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-06-23 15:15:00 | 1010.10 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-06-24 09:45:00 | 1007.65 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-06-24 13:30:00 | 1009.60 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-06-25 09:30:00 | 1014.10 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1077.90 | 2025-07-07 11:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-04 10:00:00 | 1083.00 | 2025-07-07 11:15:00 | 1069.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-07 12:30:00 | 1081.00 | 2025-07-08 10:15:00 | 1074.10 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-07-08 10:15:00 | 1079.50 | 2025-07-08 10:15:00 | 1074.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-07-15 09:45:00 | 1137.20 | 2025-07-21 13:15:00 | 1151.90 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-07-15 12:30:00 | 1135.50 | 2025-07-21 13:15:00 | 1151.90 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-07-23 12:45:00 | 1183.10 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-24 12:15:00 | 1182.70 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-24 13:00:00 | 1181.20 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-24 13:45:00 | 1181.50 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-31 13:45:00 | 1191.10 | 2025-07-31 15:15:00 | 1170.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-07 10:15:00 | 1144.80 | 2025-08-08 09:15:00 | 1087.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 12:45:00 | 1144.00 | 2025-08-08 09:15:00 | 1086.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 10:15:00 | 1144.80 | 2025-08-11 14:15:00 | 1070.00 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2025-08-07 12:45:00 | 1144.00 | 2025-08-11 14:15:00 | 1070.00 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest1 | 2025-08-29 09:15:00 | 1031.10 | 2025-09-01 09:15:00 | 1059.40 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-09-04 11:15:00 | 1080.00 | 2025-09-05 10:15:00 | 1067.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1079.20 | 2025-09-05 10:15:00 | 1067.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-08 11:15:00 | 1063.90 | 2025-09-15 12:15:00 | 1053.80 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-10-03 10:30:00 | 988.20 | 2025-10-07 10:15:00 | 1001.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-06 09:45:00 | 984.00 | 2025-10-07 10:15:00 | 1001.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-09 11:15:00 | 1006.30 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-09 12:45:00 | 1006.40 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-09 13:15:00 | 1009.10 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-09 15:00:00 | 1008.30 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-17 14:30:00 | 1027.25 | 2025-10-20 10:15:00 | 1011.85 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-20 12:30:00 | 1027.70 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-10-20 14:15:00 | 1028.85 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-10-20 15:00:00 | 1027.50 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2025-10-21 13:45:00 | 1034.15 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-10-23 09:45:00 | 1034.60 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2025-11-06 10:15:00 | 1014.00 | 2025-11-10 14:15:00 | 1032.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-11-18 12:00:00 | 983.90 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-18 13:15:00 | 984.40 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-18 14:15:00 | 984.10 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-18 14:45:00 | 984.10 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-11-26 12:45:00 | 1017.30 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-11-26 14:15:00 | 1017.10 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-27 11:45:00 | 1017.10 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-27 13:15:00 | 1016.20 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-11-28 10:15:00 | 1029.60 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1025.20 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-01 15:15:00 | 1027.20 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-02 10:45:00 | 1023.10 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1015.00 | 2025-12-03 14:15:00 | 1025.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-08 11:15:00 | 1005.10 | 2025-12-10 10:15:00 | 1023.70 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-12-08 12:00:00 | 1005.50 | 2025-12-10 10:15:00 | 1023.70 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-16 13:30:00 | 1053.10 | 2025-12-17 09:15:00 | 1040.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-16 14:45:00 | 1051.90 | 2025-12-17 09:15:00 | 1040.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-16 15:15:00 | 1052.00 | 2025-12-17 09:15:00 | 1040.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-17 14:30:00 | 1052.00 | 2025-12-18 09:15:00 | 1043.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1058.30 | 2025-12-23 09:15:00 | 1046.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-31 14:30:00 | 1053.90 | 2026-01-01 12:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-31 15:00:00 | 1053.60 | 2026-01-01 12:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1048.80 | 2026-01-01 12:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-08 10:30:00 | 1087.00 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-09 09:15:00 | 1084.10 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1085.50 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-12 10:00:00 | 1082.80 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1060.00 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-19 15:00:00 | 1065.00 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-20 09:45:00 | 1066.20 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-20 10:45:00 | 1067.00 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-21 09:15:00 | 1056.40 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-22 12:45:00 | 1077.90 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-23 09:45:00 | 1079.80 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-23 12:00:00 | 1075.00 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-01-23 13:15:00 | 1075.00 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1107.70 | 2026-02-11 10:15:00 | 1150.90 | STOP_HIT | 1.00 | 3.90% |
| BUY | retest2 | 2026-02-02 12:00:00 | 1112.00 | 2026-02-11 10:15:00 | 1150.90 | STOP_HIT | 1.00 | 3.50% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1131.10 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-02-13 14:00:00 | 1141.60 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1131.20 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-02-16 13:45:00 | 1141.60 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1118.50 | 2026-02-23 12:15:00 | 1141.50 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1092.90 | 2026-03-06 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1096.80 | 2026-03-06 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-03-06 10:30:00 | 1085.00 | 2026-03-06 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-03-13 09:30:00 | 975.60 | 2026-03-18 14:15:00 | 985.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-03-13 15:00:00 | 972.10 | 2026-03-18 14:15:00 | 985.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest1 | 2026-03-23 09:15:00 | 918.30 | 2026-03-23 14:15:00 | 872.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-23 09:15:00 | 918.30 | 2026-03-24 12:15:00 | 892.80 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2026-03-27 13:15:00 | 933.50 | 2026-03-27 14:15:00 | 909.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-04-13 15:00:00 | 994.85 | 2026-04-21 15:15:00 | 998.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-04-15 09:15:00 | 999.70 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-04-15 11:45:00 | 996.15 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-04-15 13:00:00 | 993.90 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-04-21 10:00:00 | 1014.20 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-28 11:15:00 | 958.45 | 2026-05-05 10:15:00 | 910.86 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-04-28 15:00:00 | 957.75 | 2026-05-05 10:15:00 | 914.66 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2026-04-29 09:30:00 | 958.80 | 2026-05-05 11:15:00 | 910.53 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-04-29 12:30:00 | 962.80 | 2026-05-05 11:15:00 | 909.86 | PARTIAL | 0.50 | 5.50% |
| SELL | retest2 | 2026-04-28 11:15:00 | 958.45 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2026-04-28 15:00:00 | 957.75 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2026-04-29 09:30:00 | 958.80 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-04-29 12:30:00 | 962.80 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2026-05-04 11:45:00 | 928.50 | 2026-05-06 15:15:00 | 943.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-04 14:45:00 | 928.25 | 2026-05-06 15:15:00 | 943.30 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-05-06 13:45:00 | 927.90 | 2026-05-06 15:15:00 | 943.30 | STOP_HIT | 1.00 | -1.66% |
