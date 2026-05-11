# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 237 |
| ALERT1 | 145 |
| ALERT2 | 144 |
| ALERT2_SKIP | 102 |
| ALERT3 | 345 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 140 |
| PARTIAL | 10 |
| TARGET_HIT | 7 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 156 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 117
- **Target hits / Stop hits / Partials:** 7 / 139 / 10
- **Avg / median % per leg:** -0.14% / -0.91%
- **Sum % (uncompounded):** -21.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 8 | 13.6% | 3 | 56 | 0 | -0.44% | -26.1% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.65% | 1.7% |
| BUY @ 3rd Alert (retest2) | 58 | 7 | 12.1% | 3 | 55 | 0 | -0.48% | -27.8% |
| SELL (all) | 97 | 31 | 32.0% | 4 | 83 | 10 | 0.05% | 4.7% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 5 | 0 | -0.03% | -0.2% |
| SELL @ 3rd Alert (retest2) | 92 | 27 | 29.3% | 4 | 78 | 10 | 0.05% | 4.9% |
| retest1 (combined) | 6 | 5 | 83.3% | 0 | 6 | 0 | 0.25% | 1.5% |
| retest2 (combined) | 150 | 34 | 22.7% | 7 | 133 | 10 | -0.15% | -22.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 769.10 | 764.12 | 763.90 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 11:15:00 | 760.00 | 765.21 | 765.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 13:15:00 | 755.00 | 762.33 | 763.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 769.40 | 757.67 | 759.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 769.40 | 757.67 | 759.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 769.40 | 757.67 | 759.09 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 771.95 | 760.52 | 760.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 15:15:00 | 775.00 | 768.41 | 764.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 764.30 | 767.58 | 764.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 764.30 | 767.58 | 764.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 764.30 | 767.58 | 764.74 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 12:15:00 | 905.70 | 910.87 | 911.54 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 940.75 | 913.84 | 912.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 10:15:00 | 953.65 | 934.97 | 929.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 14:15:00 | 934.95 | 941.35 | 935.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 14:15:00 | 934.95 | 941.35 | 935.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 934.95 | 941.35 | 935.18 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 13:15:00 | 1018.80 | 1033.97 | 1034.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 14:15:00 | 1015.35 | 1030.24 | 1032.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 999.20 | 989.55 | 1004.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 999.20 | 989.55 | 1004.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 999.20 | 989.55 | 1004.02 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 1046.15 | 1013.12 | 1010.39 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 11:15:00 | 989.90 | 1009.30 | 1011.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 12:15:00 | 984.65 | 1004.37 | 1009.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 1000.90 | 996.27 | 1003.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 1000.90 | 996.27 | 1003.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 1000.90 | 996.27 | 1003.04 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 1009.70 | 994.20 | 993.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 10:15:00 | 1016.15 | 1007.94 | 1002.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 12:15:00 | 1002.40 | 1007.81 | 1003.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 12:15:00 | 1002.40 | 1007.81 | 1003.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 1002.40 | 1007.81 | 1003.18 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 11:15:00 | 1034.80 | 1037.74 | 1037.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 13:15:00 | 1030.00 | 1035.64 | 1036.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 12:15:00 | 1030.00 | 1029.33 | 1032.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 1031.45 | 1030.10 | 1032.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1031.45 | 1030.10 | 1032.00 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 14:15:00 | 1038.55 | 1033.10 | 1032.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 15:15:00 | 1040.00 | 1034.48 | 1033.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 13:15:00 | 1051.35 | 1052.19 | 1046.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 14:15:00 | 1038.50 | 1049.45 | 1046.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 1038.50 | 1049.45 | 1046.00 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 11:15:00 | 1027.95 | 1047.63 | 1048.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 12:15:00 | 1013.15 | 1040.73 | 1044.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 09:15:00 | 1020.00 | 1018.41 | 1026.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 1020.00 | 1018.41 | 1026.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 1020.00 | 1018.41 | 1026.15 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 15:15:00 | 1025.00 | 1021.09 | 1020.91 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 09:15:00 | 1010.00 | 1018.87 | 1019.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 10:15:00 | 1007.25 | 1011.45 | 1014.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 11:15:00 | 993.95 | 991.45 | 997.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 11:15:00 | 993.95 | 991.45 | 997.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 11:15:00 | 993.95 | 991.45 | 997.23 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 12:15:00 | 997.65 | 991.85 | 991.22 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 15:15:00 | 989.95 | 990.79 | 990.84 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 992.55 | 991.14 | 990.99 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 12:15:00 | 989.70 | 990.91 | 990.94 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 10:15:00 | 993.35 | 991.05 | 990.91 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 14:15:00 | 989.55 | 990.70 | 990.79 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 09:15:00 | 994.40 | 991.33 | 991.05 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 12:15:00 | 987.50 | 990.46 | 990.73 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 15:15:00 | 995.00 | 991.58 | 991.16 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 10:15:00 | 987.25 | 990.64 | 990.80 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 993.30 | 991.17 | 991.02 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 13:15:00 | 989.95 | 990.74 | 990.84 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 15:15:00 | 992.90 | 991.25 | 991.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 999.00 | 992.80 | 991.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 13:15:00 | 1032.65 | 1034.44 | 1025.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 14:15:00 | 1026.75 | 1032.90 | 1025.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 1026.75 | 1032.90 | 1025.71 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 12:15:00 | 1023.15 | 1028.90 | 1029.05 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 1041.15 | 1029.65 | 1029.13 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 12:15:00 | 1024.60 | 1030.29 | 1031.00 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 10:15:00 | 1033.85 | 1031.46 | 1031.16 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 14:15:00 | 1027.65 | 1030.53 | 1030.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 11:15:00 | 1021.80 | 1027.88 | 1029.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 14:15:00 | 1014.00 | 1011.78 | 1016.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 1013.85 | 1012.07 | 1015.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 1013.85 | 1012.07 | 1015.72 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 10:15:00 | 1023.95 | 1014.00 | 1013.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 09:15:00 | 1034.20 | 1025.38 | 1020.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 1020.10 | 1032.47 | 1027.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 09:15:00 | 1020.10 | 1032.47 | 1027.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1020.10 | 1032.47 | 1027.67 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 1066.70 | 1084.61 | 1085.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 1063.65 | 1077.90 | 1082.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 15:15:00 | 1080.00 | 1077.86 | 1081.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 15:15:00 | 1080.00 | 1077.86 | 1081.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 1080.00 | 1077.86 | 1081.27 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 1082.75 | 1070.76 | 1069.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 1087.60 | 1080.17 | 1075.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 1079.15 | 1080.83 | 1077.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 12:15:00 | 1079.15 | 1080.83 | 1077.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 1079.15 | 1080.83 | 1077.04 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 11:15:00 | 1084.35 | 1092.25 | 1093.10 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 1098.90 | 1092.92 | 1092.42 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 1075.40 | 1089.25 | 1090.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 1064.50 | 1080.80 | 1086.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 14:15:00 | 1062.00 | 1059.19 | 1067.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 1061.60 | 1060.16 | 1066.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 1061.60 | 1060.16 | 1066.75 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 12:15:00 | 1069.65 | 1065.65 | 1065.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 09:15:00 | 1079.85 | 1069.14 | 1067.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 1080.60 | 1080.99 | 1075.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 15:15:00 | 1081.50 | 1087.92 | 1082.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 1081.50 | 1087.92 | 1082.72 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 1075.35 | 1080.37 | 1080.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 14:15:00 | 1073.15 | 1078.92 | 1079.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 1085.95 | 1080.02 | 1080.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 1085.95 | 1080.02 | 1080.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 1085.95 | 1080.02 | 1080.14 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 10:15:00 | 1083.80 | 1080.78 | 1080.47 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 1062.35 | 1077.15 | 1078.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 1048.60 | 1059.25 | 1066.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1015.00 | 996.11 | 1007.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1015.00 | 996.11 | 1007.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1015.00 | 996.11 | 1007.60 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 1017.15 | 1011.52 | 1010.77 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 13:15:00 | 1003.90 | 1010.57 | 1010.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 995.00 | 1007.46 | 1009.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 15:15:00 | 1002.90 | 999.06 | 1002.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 15:15:00 | 1002.90 | 999.06 | 1002.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 1002.90 | 999.06 | 1002.58 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 1007.20 | 1004.48 | 1004.27 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 14:15:00 | 1000.20 | 1003.62 | 1003.90 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 1025.00 | 1008.12 | 1005.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 11:15:00 | 1036.35 | 1016.69 | 1010.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 1059.05 | 1061.22 | 1048.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 14:15:00 | 1058.35 | 1062.87 | 1058.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1058.35 | 1062.87 | 1058.22 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 1229.50 | 1243.48 | 1243.71 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 1282.70 | 1249.31 | 1245.98 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 1230.55 | 1250.28 | 1250.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 12:15:00 | 1218.45 | 1230.35 | 1237.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 09:15:00 | 1224.95 | 1217.99 | 1224.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 1224.95 | 1217.99 | 1224.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 1224.95 | 1217.99 | 1224.16 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 1195.00 | 1175.96 | 1174.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 1201.95 | 1187.33 | 1181.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 15:15:00 | 1190.65 | 1192.13 | 1187.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 1187.00 | 1191.10 | 1187.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1187.00 | 1191.10 | 1187.03 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 13:15:00 | 1175.10 | 1185.55 | 1185.62 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 11:15:00 | 1191.30 | 1185.45 | 1185.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 13:15:00 | 1194.00 | 1187.93 | 1186.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 09:15:00 | 1186.90 | 1189.59 | 1187.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 1186.90 | 1189.59 | 1187.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 1186.90 | 1189.59 | 1187.68 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 1173.25 | 1185.06 | 1186.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1170.00 | 1179.80 | 1183.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1171.25 | 1163.54 | 1170.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1171.25 | 1163.54 | 1170.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1171.25 | 1163.54 | 1170.83 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 1181.70 | 1174.67 | 1173.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 1190.55 | 1183.38 | 1179.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 1198.00 | 1198.49 | 1191.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 1195.95 | 1197.90 | 1192.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1195.95 | 1197.90 | 1192.81 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 12:15:00 | 1193.10 | 1198.05 | 1198.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 1188.75 | 1194.91 | 1196.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 1184.95 | 1180.68 | 1184.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 15:15:00 | 1184.95 | 1180.68 | 1184.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 1184.95 | 1180.68 | 1184.78 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 1218.00 | 1188.14 | 1187.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 10:15:00 | 1222.20 | 1194.95 | 1190.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 1239.60 | 1242.84 | 1232.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 1239.60 | 1242.84 | 1232.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1239.60 | 1242.84 | 1232.18 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 14:15:00 | 1260.00 | 1269.73 | 1270.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 10:15:00 | 1254.05 | 1264.37 | 1267.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 15:15:00 | 1262.85 | 1260.19 | 1263.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 15:15:00 | 1262.85 | 1260.19 | 1263.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 15:15:00 | 1262.85 | 1260.19 | 1263.71 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 1259.25 | 1246.18 | 1244.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 11:15:00 | 1261.20 | 1249.18 | 1246.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 10:15:00 | 1242.20 | 1252.75 | 1249.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 10:15:00 | 1242.20 | 1252.75 | 1249.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 1242.20 | 1252.75 | 1249.90 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 1348.95 | 1366.77 | 1367.68 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 13:15:00 | 1374.95 | 1368.28 | 1368.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 15:15:00 | 1377.00 | 1371.20 | 1369.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 13:15:00 | 1374.95 | 1376.29 | 1373.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 15:15:00 | 1368.30 | 1374.79 | 1373.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 1368.30 | 1374.79 | 1373.07 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 09:15:00 | 1360.20 | 1371.87 | 1371.90 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 1375.00 | 1372.01 | 1371.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 09:15:00 | 1388.10 | 1375.23 | 1373.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 09:15:00 | 1361.15 | 1399.04 | 1390.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 09:15:00 | 1361.15 | 1399.04 | 1390.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 1361.15 | 1399.04 | 1390.85 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 11:15:00 | 1347.95 | 1381.17 | 1383.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 14:15:00 | 1345.90 | 1361.38 | 1366.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 11:15:00 | 1358.05 | 1357.66 | 1362.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 13:15:00 | 1350.60 | 1355.88 | 1361.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 1350.60 | 1355.88 | 1361.00 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 1368.05 | 1356.62 | 1356.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 1383.25 | 1363.28 | 1359.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 15:15:00 | 1393.00 | 1395.00 | 1384.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 1389.80 | 1393.96 | 1384.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1389.80 | 1393.96 | 1384.88 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 1375.00 | 1383.76 | 1384.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 10:15:00 | 1358.80 | 1370.76 | 1376.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 1322.00 | 1315.80 | 1335.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 1320.05 | 1316.65 | 1333.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1320.05 | 1316.65 | 1333.82 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 13:15:00 | 1270.55 | 1234.54 | 1231.33 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 1200.65 | 1241.56 | 1242.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 1190.90 | 1231.43 | 1237.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 1225.45 | 1209.24 | 1221.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 1225.45 | 1209.24 | 1221.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 1225.45 | 1209.24 | 1221.05 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 1241.75 | 1224.13 | 1224.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 11:15:00 | 1248.95 | 1232.65 | 1228.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 14:15:00 | 1238.25 | 1240.02 | 1233.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 15:15:00 | 1255.00 | 1243.02 | 1235.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1255.00 | 1243.02 | 1235.29 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 1222.75 | 1241.99 | 1243.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 1198.90 | 1231.29 | 1238.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1221.90 | 1209.80 | 1220.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1221.90 | 1209.80 | 1220.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1221.90 | 1209.80 | 1220.77 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 1230.95 | 1223.72 | 1223.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 1246.75 | 1233.59 | 1229.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 1246.85 | 1250.72 | 1242.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 1246.85 | 1250.72 | 1242.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 1246.85 | 1250.72 | 1242.39 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 1279.30 | 1291.31 | 1292.11 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 1318.00 | 1294.75 | 1293.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 10:15:00 | 1350.40 | 1305.88 | 1298.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 09:15:00 | 1315.00 | 1319.99 | 1310.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 1315.00 | 1319.99 | 1310.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 1315.00 | 1319.99 | 1310.60 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 1303.80 | 1315.33 | 1315.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 1295.20 | 1311.31 | 1313.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 12:15:00 | 1296.80 | 1294.99 | 1303.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 13:00:00 | 1296.80 | 1294.99 | 1303.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1295.95 | 1285.48 | 1294.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 1295.95 | 1285.48 | 1294.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1283.05 | 1284.99 | 1293.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 1281.00 | 1284.99 | 1293.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:45:00 | 1279.45 | 1283.31 | 1291.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 1216.95 | 1249.12 | 1265.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 1215.48 | 1249.12 | 1265.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 11:15:00 | 1247.80 | 1246.71 | 1261.09 | SL hit (close>ema200) qty=0.50 sl=1246.71 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 1272.20 | 1254.97 | 1254.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 10:15:00 | 1278.15 | 1267.67 | 1262.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 14:15:00 | 1269.75 | 1270.04 | 1265.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 15:00:00 | 1269.75 | 1270.04 | 1265.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 1260.30 | 1268.10 | 1265.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:30:00 | 1275.85 | 1269.37 | 1265.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:15:00 | 1271.60 | 1273.87 | 1270.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 13:00:00 | 1272.60 | 1270.99 | 1270.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 15:15:00 | 1271.00 | 1270.80 | 1270.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1271.00 | 1270.84 | 1270.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 1287.30 | 1270.84 | 1270.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 10:00:00 | 1271.20 | 1281.32 | 1278.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 11:30:00 | 1271.90 | 1276.50 | 1276.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 12:15:00 | 1264.50 | 1274.10 | 1275.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 12:15:00 | 1264.50 | 1274.10 | 1275.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 13:15:00 | 1257.80 | 1270.84 | 1273.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 1266.00 | 1252.94 | 1259.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 1266.00 | 1252.94 | 1259.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1266.00 | 1252.94 | 1259.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:45:00 | 1264.95 | 1252.94 | 1259.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 1275.95 | 1257.54 | 1260.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 12:00:00 | 1275.95 | 1257.54 | 1260.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 13:15:00 | 1281.15 | 1265.18 | 1263.99 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 1263.45 | 1268.84 | 1268.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 1255.95 | 1266.26 | 1267.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 10:15:00 | 1260.35 | 1258.62 | 1262.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 10:15:00 | 1260.35 | 1258.62 | 1262.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 1260.35 | 1258.62 | 1262.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 1260.35 | 1258.62 | 1262.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 1260.00 | 1258.89 | 1262.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 1268.75 | 1258.89 | 1262.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 1264.10 | 1259.93 | 1262.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:45:00 | 1265.55 | 1259.93 | 1262.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 1256.50 | 1259.25 | 1262.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:00:00 | 1256.50 | 1259.25 | 1262.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 1263.55 | 1260.11 | 1262.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 1266.65 | 1260.11 | 1262.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 1262.10 | 1260.51 | 1262.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 1266.80 | 1260.51 | 1262.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1258.00 | 1260.01 | 1261.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:30:00 | 1254.00 | 1258.88 | 1260.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:30:00 | 1254.30 | 1258.59 | 1260.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 14:00:00 | 1253.00 | 1258.43 | 1259.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 09:30:00 | 1253.30 | 1254.98 | 1257.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 1263.00 | 1256.84 | 1258.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:00:00 | 1263.00 | 1256.84 | 1258.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 1256.25 | 1256.72 | 1257.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 13:45:00 | 1253.55 | 1256.01 | 1257.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:00:00 | 1255.10 | 1255.07 | 1256.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:15:00 | 1251.00 | 1256.89 | 1257.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 15:00:00 | 1255.10 | 1254.79 | 1256.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1264.00 | 1256.63 | 1256.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 1275.45 | 1256.63 | 1256.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-17 09:15:00 | 1276.95 | 1260.69 | 1258.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 1276.95 | 1260.69 | 1258.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 1303.00 | 1269.16 | 1262.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 12:15:00 | 1292.90 | 1295.30 | 1285.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:00:00 | 1292.90 | 1295.30 | 1285.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 1280.55 | 1292.35 | 1285.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 1280.55 | 1292.35 | 1285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1284.55 | 1290.79 | 1285.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:45:00 | 1286.10 | 1290.79 | 1285.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1289.00 | 1290.43 | 1285.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1281.50 | 1290.43 | 1285.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1275.15 | 1287.37 | 1284.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1275.45 | 1287.37 | 1284.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1278.65 | 1285.63 | 1284.08 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 1275.00 | 1281.87 | 1282.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 1267.45 | 1278.99 | 1281.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 14:15:00 | 1257.00 | 1254.79 | 1262.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 1257.00 | 1254.79 | 1262.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1243.35 | 1252.22 | 1259.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1205.70 | 1244.44 | 1252.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:15:00 | 1145.41 | 1154.26 | 1182.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 1151.20 | 1149.52 | 1175.55 | SL hit (close>ema200) qty=0.50 sl=1149.52 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 1190.00 | 1176.32 | 1175.81 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1148.95 | 1176.40 | 1178.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1140.85 | 1169.29 | 1174.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 15:15:00 | 1172.00 | 1162.95 | 1169.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 15:15:00 | 1172.00 | 1162.95 | 1169.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 1172.00 | 1162.95 | 1169.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 1152.15 | 1162.95 | 1169.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1141.00 | 1158.56 | 1166.75 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 1182.00 | 1170.14 | 1170.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1267.20 | 1194.84 | 1181.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1270.10 | 1289.90 | 1269.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 1270.10 | 1289.90 | 1269.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1270.10 | 1289.90 | 1269.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1265.00 | 1289.90 | 1269.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1271.55 | 1286.23 | 1270.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 11:15:00 | 1277.20 | 1286.23 | 1270.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 1254.70 | 1273.74 | 1268.82 | SL hit (close<static) qty=1.00 sl=1261.20 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1251.80 | 1263.87 | 1265.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 13:15:00 | 1247.55 | 1258.91 | 1262.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 1242.25 | 1235.40 | 1244.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 1242.25 | 1235.40 | 1244.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1242.25 | 1235.40 | 1244.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 1247.95 | 1235.40 | 1244.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1248.80 | 1238.08 | 1244.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:00:00 | 1248.80 | 1238.08 | 1244.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 1244.50 | 1239.37 | 1244.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 13:45:00 | 1238.65 | 1240.31 | 1244.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 1253.05 | 1242.18 | 1244.14 | SL hit (close>static) qty=1.00 sl=1250.90 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 11:15:00 | 1257.20 | 1247.32 | 1246.27 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 1234.20 | 1243.52 | 1244.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 1226.45 | 1238.47 | 1241.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 13:15:00 | 1236.25 | 1234.32 | 1238.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 13:15:00 | 1236.25 | 1234.32 | 1238.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 1236.25 | 1234.32 | 1238.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 1239.00 | 1234.32 | 1238.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1236.30 | 1234.87 | 1237.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 1238.75 | 1234.87 | 1237.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1239.05 | 1235.70 | 1238.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1239.05 | 1235.70 | 1238.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1234.00 | 1235.36 | 1237.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 11:15:00 | 1232.00 | 1235.36 | 1237.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:15:00 | 1233.85 | 1234.83 | 1237.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:00:00 | 1233.00 | 1234.47 | 1236.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1231.70 | 1201.03 | 1199.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 1231.70 | 1201.03 | 1199.32 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 12:15:00 | 1194.95 | 1203.22 | 1203.30 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 15:15:00 | 1206.95 | 1203.60 | 1203.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1211.05 | 1205.09 | 1204.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 1221.25 | 1224.93 | 1219.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:00:00 | 1221.25 | 1224.93 | 1219.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1223.65 | 1224.67 | 1220.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 1219.80 | 1224.67 | 1220.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1238.10 | 1245.01 | 1239.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1238.10 | 1245.01 | 1239.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1228.05 | 1241.62 | 1238.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 1228.05 | 1241.62 | 1238.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1228.05 | 1238.90 | 1237.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:15:00 | 1222.85 | 1238.90 | 1237.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 1224.60 | 1236.04 | 1236.51 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 15:15:00 | 1250.00 | 1236.74 | 1236.62 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 1224.30 | 1234.25 | 1235.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 1220.35 | 1227.98 | 1231.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 10:15:00 | 1242.00 | 1230.78 | 1232.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 10:15:00 | 1242.00 | 1230.78 | 1232.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1242.00 | 1230.78 | 1232.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:00:00 | 1242.00 | 1230.78 | 1232.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1242.00 | 1233.02 | 1233.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:45:00 | 1246.25 | 1233.02 | 1233.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 1233.00 | 1232.54 | 1232.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:30:00 | 1238.45 | 1232.54 | 1232.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1223.65 | 1230.76 | 1232.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:15:00 | 1232.50 | 1230.76 | 1232.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1232.50 | 1231.11 | 1232.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1236.20 | 1231.11 | 1232.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1229.65 | 1230.82 | 1231.89 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 1239.75 | 1232.60 | 1232.60 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 1228.75 | 1232.99 | 1233.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 1220.35 | 1230.46 | 1232.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 1234.95 | 1225.34 | 1228.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 1234.95 | 1225.34 | 1228.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1234.95 | 1225.34 | 1228.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 1234.95 | 1225.34 | 1228.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1239.50 | 1228.17 | 1229.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 1241.35 | 1228.17 | 1229.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 1250.30 | 1232.60 | 1231.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 1259.00 | 1250.70 | 1247.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 14:15:00 | 1256.50 | 1257.46 | 1252.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 14:15:00 | 1256.50 | 1257.46 | 1252.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1256.50 | 1257.46 | 1252.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 1256.50 | 1257.46 | 1252.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1254.90 | 1256.95 | 1252.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 1268.00 | 1256.95 | 1252.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 14:30:00 | 1258.00 | 1260.07 | 1256.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 1240.10 | 1255.40 | 1255.15 | SL hit (close<static) qty=1.00 sl=1250.10 alert=retest2 |

### Cycle 96 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 10:15:00 | 1241.00 | 1252.52 | 1253.86 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1270.00 | 1251.73 | 1250.69 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 1259.40 | 1264.71 | 1264.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1238.90 | 1253.95 | 1258.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 09:15:00 | 1239.05 | 1237.03 | 1245.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 1239.05 | 1237.03 | 1245.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1239.05 | 1237.03 | 1245.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:45:00 | 1245.60 | 1237.03 | 1245.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 1221.45 | 1227.29 | 1237.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 13:45:00 | 1229.80 | 1227.29 | 1237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1249.60 | 1229.48 | 1235.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 1235.95 | 1230.96 | 1236.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:00:00 | 1236.90 | 1230.96 | 1236.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1238.15 | 1229.07 | 1228.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1238.15 | 1229.07 | 1228.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 11:15:00 | 1245.80 | 1233.12 | 1230.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 1230.60 | 1234.26 | 1231.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 1230.60 | 1234.26 | 1231.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1230.60 | 1234.26 | 1231.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 1228.95 | 1234.26 | 1231.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1232.65 | 1233.93 | 1231.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 1222.00 | 1233.93 | 1231.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1222.65 | 1231.68 | 1231.12 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 1221.80 | 1229.70 | 1230.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 1219.80 | 1227.72 | 1229.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 15:15:00 | 1206.90 | 1202.84 | 1211.17 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 1194.10 | 1201.70 | 1209.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:30:00 | 1194.75 | 1201.09 | 1208.87 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 11:30:00 | 1192.00 | 1199.85 | 1207.60 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 12:45:00 | 1195.90 | 1198.23 | 1206.16 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1176.10 | 1182.84 | 1190.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-19 14:15:00 | 1187.75 | 1181.31 | 1186.40 | SL hit (close>ema400) qty=1.00 sl=1186.40 alert=retest1 |

### Cycle 101 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 1217.95 | 1188.18 | 1186.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 1232.90 | 1197.12 | 1190.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 12:15:00 | 1263.05 | 1265.29 | 1249.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 13:00:00 | 1263.05 | 1265.29 | 1249.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1270.00 | 1276.13 | 1269.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:45:00 | 1271.30 | 1276.13 | 1269.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1267.95 | 1274.49 | 1269.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 1265.00 | 1274.49 | 1269.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1270.00 | 1273.60 | 1269.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1282.00 | 1273.60 | 1269.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 1278.00 | 1276.88 | 1275.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 13:15:00 | 1266.85 | 1274.00 | 1274.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 1266.85 | 1274.00 | 1274.09 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 1283.25 | 1274.75 | 1274.07 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 1269.90 | 1275.39 | 1275.41 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 1286.15 | 1276.51 | 1275.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1317.60 | 1288.41 | 1281.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 1360.30 | 1363.02 | 1343.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:45:00 | 1358.40 | 1363.02 | 1343.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 1349.35 | 1355.20 | 1345.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 1358.00 | 1352.27 | 1345.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 1341.60 | 1350.14 | 1345.25 | SL hit (close<static) qty=1.00 sl=1343.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1326.65 | 1341.04 | 1342.59 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 1364.15 | 1347.13 | 1344.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 1372.40 | 1356.15 | 1349.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 1314.75 | 1350.09 | 1348.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 1314.75 | 1350.09 | 1348.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1314.75 | 1350.09 | 1348.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 1314.75 | 1350.09 | 1348.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 1310.55 | 1342.18 | 1344.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 12:15:00 | 1309.80 | 1330.63 | 1338.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 1263.65 | 1255.84 | 1267.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 1263.65 | 1255.84 | 1267.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1263.65 | 1255.84 | 1267.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:45:00 | 1245.00 | 1256.33 | 1264.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 1240.00 | 1256.33 | 1264.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 1245.50 | 1253.13 | 1261.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 1259.95 | 1242.69 | 1242.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 1259.95 | 1242.69 | 1242.29 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 1222.40 | 1243.29 | 1244.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 11:15:00 | 1208.00 | 1232.37 | 1238.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 1227.90 | 1218.48 | 1227.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 1227.90 | 1218.48 | 1227.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1227.90 | 1218.48 | 1227.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 1227.90 | 1218.48 | 1227.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1236.55 | 1222.09 | 1228.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 1236.55 | 1222.09 | 1228.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 1249.10 | 1227.50 | 1230.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 1249.10 | 1227.50 | 1230.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 1251.35 | 1236.19 | 1234.15 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 1230.15 | 1235.74 | 1236.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 1226.00 | 1233.79 | 1235.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 1232.15 | 1230.13 | 1232.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 14:15:00 | 1232.15 | 1230.13 | 1232.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1232.15 | 1230.13 | 1232.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 1232.15 | 1230.13 | 1232.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1234.00 | 1230.91 | 1232.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 1210.85 | 1230.91 | 1232.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 15:15:00 | 1222.00 | 1219.60 | 1222.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1232.40 | 1223.48 | 1223.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 1232.40 | 1223.48 | 1223.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 1237.30 | 1228.77 | 1226.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 12:15:00 | 1227.95 | 1228.68 | 1226.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 12:15:00 | 1227.95 | 1228.68 | 1226.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 1227.95 | 1228.68 | 1226.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:45:00 | 1226.90 | 1228.68 | 1226.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 1228.50 | 1228.64 | 1226.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:45:00 | 1226.45 | 1228.64 | 1226.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 1224.15 | 1227.74 | 1226.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 1224.15 | 1227.74 | 1226.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 1221.00 | 1226.39 | 1225.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:30:00 | 1228.10 | 1226.78 | 1226.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:15:00 | 1226.25 | 1228.29 | 1227.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:15:00 | 1226.30 | 1227.40 | 1226.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:30:00 | 1229.15 | 1227.93 | 1227.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1229.10 | 1229.10 | 1227.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 1230.60 | 1229.10 | 1227.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 1221.30 | 1227.54 | 1227.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:00:00 | 1221.30 | 1227.54 | 1227.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-11 13:15:00 | 1221.20 | 1226.27 | 1226.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 1221.20 | 1226.27 | 1226.81 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 1236.00 | 1226.94 | 1226.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 1252.35 | 1234.37 | 1230.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 12:15:00 | 1267.00 | 1267.03 | 1258.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 13:00:00 | 1267.00 | 1267.03 | 1258.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1269.20 | 1278.78 | 1272.95 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 1261.05 | 1275.02 | 1276.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 10:15:00 | 1257.45 | 1262.74 | 1268.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 11:15:00 | 1245.60 | 1245.33 | 1254.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 11:45:00 | 1245.05 | 1245.33 | 1254.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1245.95 | 1236.46 | 1242.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 1243.00 | 1236.46 | 1242.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1255.90 | 1240.35 | 1243.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 1255.90 | 1240.35 | 1243.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1249.75 | 1242.23 | 1244.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:45:00 | 1240.95 | 1241.84 | 1243.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 1230.00 | 1237.08 | 1240.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 10:15:00 | 1251.45 | 1233.63 | 1233.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 1251.45 | 1233.63 | 1233.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 1257.60 | 1238.42 | 1235.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 17:15:00 | 1252.40 | 1256.82 | 1247.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1252.40 | 1256.82 | 1247.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1252.40 | 1256.82 | 1247.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1252.40 | 1256.82 | 1247.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1235.00 | 1252.46 | 1246.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1235.00 | 1252.46 | 1246.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1209.70 | 1243.91 | 1243.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1209.70 | 1243.91 | 1243.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1220.10 | 1239.14 | 1240.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 1196.65 | 1216.63 | 1227.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1220.00 | 1203.60 | 1213.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 1220.00 | 1203.60 | 1213.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1220.00 | 1203.60 | 1213.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 1220.00 | 1203.60 | 1213.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1237.50 | 1210.38 | 1215.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 1233.60 | 1210.38 | 1215.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 1240.55 | 1220.07 | 1219.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 1245.65 | 1228.51 | 1223.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 12:15:00 | 1289.65 | 1292.29 | 1276.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 13:00:00 | 1289.65 | 1292.29 | 1276.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 1290.65 | 1293.57 | 1285.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:45:00 | 1290.30 | 1293.57 | 1285.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 1284.00 | 1290.48 | 1285.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 1259.60 | 1290.48 | 1285.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1251.70 | 1282.72 | 1282.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 1251.95 | 1282.72 | 1282.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1259.60 | 1278.10 | 1280.59 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 1285.80 | 1265.36 | 1264.59 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 11:15:00 | 1254.95 | 1267.97 | 1268.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 13:15:00 | 1253.45 | 1264.31 | 1266.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 15:15:00 | 1264.00 | 1263.17 | 1265.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:15:00 | 1272.45 | 1263.17 | 1265.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1263.25 | 1263.18 | 1265.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 12:30:00 | 1259.05 | 1261.68 | 1263.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 13:00:00 | 1259.00 | 1261.68 | 1263.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 13:30:00 | 1254.35 | 1261.31 | 1263.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 14:45:00 | 1257.50 | 1262.17 | 1263.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1260.25 | 1261.79 | 1263.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 1258.55 | 1261.79 | 1263.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:00:00 | 1259.35 | 1261.29 | 1262.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 1256.25 | 1259.55 | 1261.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 1255.05 | 1251.48 | 1255.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1258.00 | 1252.78 | 1255.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:45:00 | 1258.20 | 1252.78 | 1255.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1242.90 | 1250.81 | 1254.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-29 15:15:00 | 1270.00 | 1256.90 | 1256.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 15:15:00 | 1270.00 | 1256.90 | 1256.13 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 1250.00 | 1255.02 | 1255.59 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1276.05 | 1259.43 | 1257.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1297.20 | 1275.10 | 1267.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 13:15:00 | 1312.75 | 1312.79 | 1298.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 13:45:00 | 1310.10 | 1312.79 | 1298.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 1309.70 | 1312.09 | 1300.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:30:00 | 1318.80 | 1313.28 | 1302.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 10:15:00 | 1317.85 | 1325.89 | 1320.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 1302.75 | 1315.94 | 1317.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1302.75 | 1315.94 | 1317.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 1288.20 | 1310.39 | 1314.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 09:15:00 | 1293.00 | 1287.36 | 1296.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 1293.00 | 1287.36 | 1296.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1293.00 | 1287.36 | 1296.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 1290.35 | 1287.36 | 1296.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1297.80 | 1289.44 | 1297.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 1297.80 | 1289.44 | 1297.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1294.55 | 1290.47 | 1296.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:45:00 | 1291.90 | 1290.47 | 1296.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:15:00 | 1292.00 | 1292.41 | 1296.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:30:00 | 1290.45 | 1291.11 | 1292.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 1298.90 | 1293.82 | 1293.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 1298.90 | 1293.82 | 1293.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1309.50 | 1298.11 | 1295.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 1300.00 | 1301.55 | 1298.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:15:00 | 1294.65 | 1301.55 | 1298.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1309.00 | 1303.04 | 1299.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:30:00 | 1300.70 | 1303.04 | 1299.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1298.05 | 1302.04 | 1299.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 1298.55 | 1302.04 | 1299.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1291.75 | 1299.99 | 1298.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:30:00 | 1294.00 | 1298.45 | 1297.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 1292.70 | 1297.30 | 1297.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 1288.85 | 1295.61 | 1296.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 1288.80 | 1288.31 | 1291.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 1288.80 | 1288.31 | 1291.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1288.80 | 1288.31 | 1291.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:30:00 | 1289.40 | 1288.31 | 1291.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1287.75 | 1288.20 | 1291.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 1287.95 | 1288.20 | 1291.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1287.00 | 1287.92 | 1290.09 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 13:15:00 | 1294.00 | 1291.52 | 1291.38 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 1286.00 | 1290.42 | 1290.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 1279.50 | 1288.01 | 1289.70 | Break + close below crossover candle low |

### Cycle 131 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 1307.95 | 1292.00 | 1291.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 14:15:00 | 1313.50 | 1300.07 | 1295.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 15:15:00 | 1302.20 | 1302.83 | 1299.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 15:15:00 | 1302.20 | 1302.83 | 1299.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1302.20 | 1302.83 | 1299.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 10:00:00 | 1314.90 | 1305.25 | 1301.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 11:30:00 | 1309.00 | 1307.48 | 1305.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 1308.95 | 1307.48 | 1305.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:00:00 | 1307.80 | 1307.54 | 1305.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1301.45 | 1306.32 | 1305.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 1301.25 | 1306.32 | 1305.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 14:15:00 | 1297.10 | 1304.48 | 1304.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 1297.10 | 1304.48 | 1304.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 1274.65 | 1291.43 | 1297.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 09:15:00 | 1284.90 | 1278.23 | 1284.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 1284.90 | 1278.23 | 1284.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1284.90 | 1278.23 | 1284.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 1292.50 | 1278.23 | 1284.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1288.80 | 1280.34 | 1285.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 1288.80 | 1280.34 | 1285.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1295.95 | 1283.46 | 1286.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 1295.95 | 1283.46 | 1286.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 1306.05 | 1291.43 | 1289.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 1311.60 | 1295.47 | 1291.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 13:15:00 | 1304.00 | 1304.24 | 1298.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 14:00:00 | 1304.00 | 1304.24 | 1298.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 1299.95 | 1303.38 | 1298.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 1299.95 | 1303.38 | 1298.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 1300.50 | 1302.80 | 1298.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 1302.55 | 1302.80 | 1298.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 14:15:00 | 1328.60 | 1342.09 | 1342.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 1328.60 | 1342.09 | 1342.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1322.50 | 1336.40 | 1339.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 1288.10 | 1287.53 | 1301.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 12:00:00 | 1288.10 | 1287.53 | 1301.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1292.00 | 1288.95 | 1298.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 1297.05 | 1288.95 | 1298.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1285.50 | 1288.08 | 1296.67 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1310.10 | 1298.64 | 1297.11 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 1298.35 | 1300.63 | 1300.88 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 1307.10 | 1301.92 | 1301.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 1311.10 | 1303.76 | 1302.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1310.65 | 1311.87 | 1307.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1310.65 | 1311.87 | 1307.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1310.65 | 1311.87 | 1307.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:45:00 | 1309.40 | 1311.87 | 1307.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1321.00 | 1317.01 | 1310.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 1318.85 | 1317.01 | 1310.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1312.60 | 1316.94 | 1312.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 1307.95 | 1316.94 | 1312.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1313.00 | 1316.15 | 1312.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:45:00 | 1322.00 | 1313.73 | 1312.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 1327.70 | 1317.18 | 1314.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 1321.65 | 1316.32 | 1314.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 10:30:00 | 1317.60 | 1321.00 | 1318.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1318.60 | 1320.52 | 1318.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:30:00 | 1321.60 | 1320.52 | 1318.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 1317.15 | 1319.85 | 1318.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 1317.15 | 1319.85 | 1318.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 1298.90 | 1315.66 | 1316.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 1298.90 | 1315.66 | 1316.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1265.10 | 1301.58 | 1309.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 1293.70 | 1293.66 | 1301.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 1293.70 | 1293.66 | 1301.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1271.00 | 1289.24 | 1298.49 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 1311.00 | 1289.41 | 1288.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1319.00 | 1304.95 | 1298.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 14:15:00 | 1353.00 | 1354.55 | 1337.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 15:00:00 | 1353.00 | 1354.55 | 1337.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 1354.00 | 1356.95 | 1344.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 1354.00 | 1356.95 | 1344.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1353.70 | 1357.67 | 1349.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:00:00 | 1380.65 | 1359.81 | 1354.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 14:30:00 | 1379.00 | 1371.95 | 1363.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 1380.50 | 1371.51 | 1364.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 1395.45 | 1380.68 | 1369.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1376.60 | 1381.32 | 1372.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:45:00 | 1370.20 | 1381.32 | 1372.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1404.00 | 1385.85 | 1375.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1407.20 | 1389.68 | 1378.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 1358.50 | 1387.09 | 1384.31 | SL hit (close<static) qty=1.00 sl=1374.30 alert=retest2 |

### Cycle 140 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 1349.95 | 1379.66 | 1381.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1334.00 | 1365.78 | 1374.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 1363.00 | 1359.43 | 1368.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:15:00 | 1336.00 | 1359.43 | 1368.64 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1367.00 | 1352.98 | 1361.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 1367.00 | 1352.98 | 1361.55 | SL hit (close>ema400) qty=1.00 sl=1361.55 alert=retest1 |

### Cycle 141 — BUY (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 10:15:00 | 1364.45 | 1337.79 | 1337.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1374.55 | 1366.77 | 1357.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1391.15 | 1395.52 | 1382.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1391.15 | 1395.52 | 1382.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1388.30 | 1394.08 | 1382.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 1377.80 | 1394.08 | 1382.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 1365.30 | 1387.24 | 1381.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 1365.30 | 1387.24 | 1381.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1382.30 | 1386.25 | 1381.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 1357.05 | 1386.25 | 1381.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1403.60 | 1389.72 | 1383.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:45:00 | 1377.95 | 1389.72 | 1383.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1391.90 | 1392.87 | 1386.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 1394.45 | 1392.87 | 1386.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1389.70 | 1392.23 | 1386.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 1389.75 | 1392.23 | 1386.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1388.05 | 1391.40 | 1386.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:00:00 | 1388.05 | 1391.40 | 1386.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1380.00 | 1389.12 | 1386.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1380.00 | 1389.12 | 1386.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 1384.65 | 1388.22 | 1385.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:30:00 | 1373.00 | 1388.22 | 1385.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 1392.95 | 1389.17 | 1386.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:15:00 | 1389.40 | 1389.17 | 1386.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1389.40 | 1389.21 | 1386.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1396.10 | 1389.21 | 1386.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-03 12:15:00 | 1535.71 | 1488.53 | 1470.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 1562.20 | 1576.06 | 1577.48 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 1595.00 | 1578.50 | 1578.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 09:15:00 | 1603.50 | 1583.50 | 1580.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 1568.60 | 1580.52 | 1579.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 1568.60 | 1580.52 | 1579.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1568.60 | 1580.52 | 1579.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 1568.60 | 1580.52 | 1579.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 1563.65 | 1577.14 | 1577.87 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 1581.00 | 1567.21 | 1566.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1592.00 | 1574.21 | 1570.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 10:15:00 | 1676.35 | 1677.78 | 1655.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:00:00 | 1676.35 | 1677.78 | 1655.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1662.80 | 1675.36 | 1661.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 1662.80 | 1675.36 | 1661.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1664.55 | 1673.20 | 1661.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 11:30:00 | 1685.00 | 1674.92 | 1665.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 1645.90 | 1669.65 | 1668.87 | SL hit (close<static) qty=1.00 sl=1655.55 alert=retest2 |

### Cycle 146 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1652.60 | 1666.24 | 1667.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1633.95 | 1651.93 | 1659.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 1670.20 | 1644.54 | 1650.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 1670.20 | 1644.54 | 1650.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1670.20 | 1644.54 | 1650.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 1670.20 | 1644.54 | 1650.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1674.75 | 1650.58 | 1652.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 1676.40 | 1650.58 | 1652.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1689.20 | 1658.31 | 1655.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 12:15:00 | 1698.00 | 1666.25 | 1659.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 1680.10 | 1683.57 | 1675.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 1680.10 | 1683.57 | 1675.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1680.10 | 1683.57 | 1675.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:45:00 | 1679.40 | 1683.57 | 1675.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1688.35 | 1684.53 | 1677.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 1678.45 | 1684.53 | 1677.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 1676.20 | 1683.02 | 1678.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 1676.20 | 1683.02 | 1678.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 1680.00 | 1682.42 | 1678.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 1664.70 | 1682.42 | 1678.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1679.00 | 1681.74 | 1678.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 1686.00 | 1678.41 | 1677.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 1652.50 | 1674.46 | 1676.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 1652.50 | 1674.46 | 1676.40 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 1680.00 | 1670.37 | 1669.87 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1568.35 | 1649.97 | 1660.64 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 1690.95 | 1654.48 | 1650.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 1742.60 | 1691.17 | 1673.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 1679.70 | 1692.37 | 1676.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 13:15:00 | 1679.70 | 1692.37 | 1676.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 1679.70 | 1692.37 | 1676.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:30:00 | 1681.10 | 1692.37 | 1676.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 1690.70 | 1692.03 | 1678.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:30:00 | 1692.75 | 1692.03 | 1678.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 1675.00 | 1688.67 | 1681.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:00:00 | 1675.00 | 1688.67 | 1681.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 1699.55 | 1690.85 | 1682.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 14:00:00 | 1705.05 | 1693.69 | 1684.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 1726.50 | 1696.28 | 1687.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 12:15:00 | 1798.80 | 1816.89 | 1819.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 1798.80 | 1816.89 | 1819.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1785.00 | 1798.74 | 1805.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 1789.40 | 1789.13 | 1798.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 1789.40 | 1789.13 | 1798.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 1789.40 | 1789.13 | 1798.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:45:00 | 1796.40 | 1789.13 | 1798.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1821.80 | 1795.89 | 1798.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 1818.70 | 1795.89 | 1798.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1809.20 | 1798.55 | 1799.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 1820.70 | 1798.55 | 1799.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 1797.50 | 1798.59 | 1799.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 1800.00 | 1798.59 | 1799.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 1798.10 | 1798.49 | 1799.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:30:00 | 1796.00 | 1798.49 | 1799.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1798.90 | 1798.57 | 1799.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:45:00 | 1798.50 | 1798.57 | 1799.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 1812.00 | 1801.26 | 1800.64 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 10:15:00 | 1793.30 | 1799.45 | 1799.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 1790.30 | 1797.62 | 1799.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 12:15:00 | 1800.00 | 1798.10 | 1799.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 12:15:00 | 1800.00 | 1798.10 | 1799.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 1800.00 | 1798.10 | 1799.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 1800.00 | 1798.10 | 1799.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 1772.60 | 1793.00 | 1796.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:30:00 | 1799.90 | 1793.00 | 1796.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 1783.50 | 1784.17 | 1790.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:30:00 | 1794.90 | 1784.17 | 1790.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1779.20 | 1770.56 | 1780.06 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 1844.00 | 1790.56 | 1783.29 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1776.30 | 1808.78 | 1809.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1768.70 | 1800.77 | 1805.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 10:15:00 | 1786.10 | 1780.26 | 1789.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 10:15:00 | 1786.10 | 1780.26 | 1789.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1786.10 | 1780.26 | 1789.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 1786.10 | 1780.26 | 1789.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 11:15:00 | 1791.10 | 1782.43 | 1789.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:30:00 | 1793.50 | 1782.43 | 1789.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 1785.20 | 1782.98 | 1788.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 12:30:00 | 1795.10 | 1782.98 | 1788.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 1779.80 | 1781.87 | 1787.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 14:45:00 | 1784.80 | 1781.87 | 1787.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1782.10 | 1781.62 | 1786.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 11:15:00 | 1771.10 | 1781.27 | 1785.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 09:15:00 | 1797.80 | 1778.93 | 1781.31 | SL hit (close>static) qty=1.00 sl=1795.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 1796.00 | 1784.28 | 1783.44 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 1779.50 | 1782.56 | 1782.76 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 1792.20 | 1783.67 | 1783.14 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 1775.50 | 1783.73 | 1784.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 1770.00 | 1779.37 | 1782.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 1762.00 | 1756.14 | 1764.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1762.00 | 1756.14 | 1764.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1762.00 | 1756.14 | 1764.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1761.00 | 1756.14 | 1764.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1756.90 | 1756.29 | 1763.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 1762.00 | 1756.29 | 1763.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1754.40 | 1745.03 | 1753.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1752.50 | 1745.03 | 1753.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1753.20 | 1746.66 | 1753.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 1744.40 | 1745.01 | 1752.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1822.30 | 1746.90 | 1740.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1822.30 | 1746.90 | 1740.38 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 1744.00 | 1755.60 | 1756.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1723.70 | 1749.22 | 1753.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1745.50 | 1733.96 | 1741.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1745.50 | 1733.96 | 1741.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1745.50 | 1733.96 | 1741.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 1745.50 | 1733.96 | 1741.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1750.00 | 1737.17 | 1741.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 1750.00 | 1737.17 | 1741.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1756.30 | 1740.99 | 1743.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 1761.80 | 1740.99 | 1743.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 1772.60 | 1747.32 | 1745.95 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1737.70 | 1749.20 | 1750.66 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1785.80 | 1753.30 | 1751.04 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 1740.70 | 1754.23 | 1755.63 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1772.00 | 1753.79 | 1751.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 1792.80 | 1761.59 | 1755.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 1830.10 | 1832.51 | 1816.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 11:00:00 | 1830.10 | 1832.51 | 1816.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1813.50 | 1826.77 | 1816.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1813.50 | 1826.77 | 1816.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1811.40 | 1823.70 | 1815.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1811.40 | 1823.70 | 1815.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1820.20 | 1822.18 | 1816.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1846.90 | 1822.18 | 1816.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1847.90 | 1827.33 | 1819.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 1886.50 | 1847.31 | 1831.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 1878.70 | 1900.87 | 1902.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 1878.70 | 1900.87 | 1902.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 1872.20 | 1891.94 | 1897.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1890.50 | 1885.11 | 1891.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1890.50 | 1885.11 | 1891.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1890.50 | 1885.11 | 1891.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1890.50 | 1885.11 | 1891.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1874.20 | 1882.92 | 1890.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 1863.00 | 1878.94 | 1887.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1903.00 | 1883.75 | 1889.20 | SL hit (close>static) qty=1.00 sl=1893.80 alert=retest2 |

### Cycle 169 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1926.10 | 1898.64 | 1895.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1987.80 | 1922.56 | 1907.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 2063.40 | 2068.53 | 2033.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 2063.40 | 2068.53 | 2033.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2036.10 | 2062.60 | 2036.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 2036.10 | 2062.60 | 2036.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2041.70 | 2058.42 | 2037.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 2045.90 | 2058.42 | 2037.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:45:00 | 2044.00 | 2050.66 | 2036.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 14:15:00 | 2250.49 | 2115.81 | 2069.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 2054.20 | 2104.15 | 2109.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 2026.90 | 2088.70 | 2101.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2075.70 | 2032.67 | 2061.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2075.70 | 2032.67 | 2061.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2075.70 | 2032.67 | 2061.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 2075.70 | 2032.67 | 2061.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2059.00 | 2037.93 | 2061.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 2044.20 | 2041.95 | 2060.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 2009.90 | 1984.27 | 1980.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 10:15:00 | 2009.90 | 1984.27 | 1980.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 2038.70 | 2007.75 | 1995.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 14:15:00 | 2019.30 | 2019.70 | 2007.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 2019.30 | 2019.70 | 2007.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1990.00 | 2013.29 | 2006.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 1990.00 | 2013.29 | 2006.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1997.10 | 2010.05 | 2005.51 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 1987.70 | 2000.82 | 2001.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1965.30 | 1985.06 | 1993.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1972.70 | 1967.12 | 1979.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1972.70 | 1967.12 | 1979.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1972.70 | 1967.12 | 1979.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1949.20 | 1965.23 | 1973.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:15:00 | 1940.70 | 1960.82 | 1969.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1934.60 | 1959.48 | 1967.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 1935.30 | 1957.94 | 1965.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1965.70 | 1958.84 | 1963.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1962.00 | 1958.84 | 1963.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1933.70 | 1953.81 | 1961.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1929.90 | 1948.93 | 1958.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 1971.10 | 1944.55 | 1953.13 | SL hit (close>static) qty=1.00 sl=1968.70 alert=retest2 |

### Cycle 173 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 1971.50 | 1957.92 | 1957.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 1996.30 | 1974.55 | 1966.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1980.00 | 1992.34 | 1985.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1980.00 | 1992.34 | 1985.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1980.00 | 1992.34 | 1985.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1980.00 | 1992.34 | 1985.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1980.00 | 1989.87 | 1984.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1982.40 | 1989.87 | 1984.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1994.20 | 1987.88 | 1984.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1999.60 | 1989.77 | 1986.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 1972.50 | 1986.11 | 1986.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 1972.50 | 1986.11 | 1986.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1968.60 | 1977.44 | 1981.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1960.90 | 1955.38 | 1966.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1960.90 | 1955.38 | 1966.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1960.90 | 1955.38 | 1966.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 1968.30 | 1955.38 | 1966.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1946.00 | 1953.51 | 1964.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1943.00 | 1953.51 | 1964.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 1944.50 | 1951.61 | 1962.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 1941.00 | 1951.86 | 1961.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 1935.70 | 1946.39 | 1958.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1944.60 | 1938.68 | 1948.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1944.20 | 1938.68 | 1948.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1947.60 | 1940.46 | 1948.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1947.60 | 1940.46 | 1948.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1952.00 | 1942.77 | 1948.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:30:00 | 1950.50 | 1942.77 | 1948.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1942.70 | 1942.76 | 1948.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:45:00 | 1961.00 | 1942.76 | 1948.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1941.10 | 1942.42 | 1947.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1861.30 | 1942.42 | 1947.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1845.85 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1847.27 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1843.95 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1838.91 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 1768.23 | 1814.28 | 1846.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-07 09:15:00 | 1748.70 | 1776.78 | 1809.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 175 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 1797.50 | 1769.50 | 1768.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 1809.80 | 1777.56 | 1771.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1811.50 | 1814.47 | 1802.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1811.50 | 1814.47 | 1802.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1814.00 | 1817.50 | 1808.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 1831.60 | 1818.81 | 1810.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1823.60 | 1835.63 | 1836.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1823.60 | 1835.63 | 1836.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1820.70 | 1832.64 | 1834.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 1829.00 | 1824.84 | 1829.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 1829.00 | 1824.84 | 1829.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1829.00 | 1824.84 | 1829.32 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 1848.00 | 1833.04 | 1832.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 1849.70 | 1836.37 | 1833.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1817.00 | 1835.13 | 1834.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1817.00 | 1835.13 | 1834.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1817.00 | 1835.13 | 1834.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1830.90 | 1835.13 | 1834.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1803.60 | 1828.82 | 1831.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 1793.70 | 1821.80 | 1827.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1768.80 | 1766.58 | 1783.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 1770.10 | 1766.58 | 1783.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1767.30 | 1760.09 | 1772.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1767.30 | 1760.09 | 1772.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1777.90 | 1764.45 | 1772.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1777.90 | 1764.45 | 1772.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1769.70 | 1765.50 | 1772.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:15:00 | 1786.50 | 1765.50 | 1772.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1790.10 | 1770.42 | 1773.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1790.10 | 1770.42 | 1773.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1788.40 | 1774.02 | 1775.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 1782.90 | 1774.02 | 1775.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 1782.90 | 1775.79 | 1775.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1782.90 | 1775.79 | 1775.75 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 1772.40 | 1775.74 | 1775.82 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1795.00 | 1775.81 | 1775.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 1807.40 | 1784.73 | 1779.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1797.40 | 1798.54 | 1790.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1797.40 | 1798.54 | 1790.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1797.40 | 1798.54 | 1790.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1797.40 | 1798.54 | 1790.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1798.60 | 1812.97 | 1805.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 1800.60 | 1812.97 | 1805.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1797.30 | 1809.83 | 1804.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1797.30 | 1809.83 | 1804.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 1785.00 | 1799.28 | 1800.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 1766.30 | 1792.69 | 1797.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 14:15:00 | 1782.50 | 1778.39 | 1787.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 1782.50 | 1778.39 | 1787.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1782.50 | 1778.39 | 1787.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 1789.90 | 1778.39 | 1787.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1780.40 | 1778.79 | 1786.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1785.00 | 1778.79 | 1786.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1771.00 | 1777.23 | 1785.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 1763.40 | 1772.15 | 1781.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 1763.50 | 1770.89 | 1773.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 1754.10 | 1770.89 | 1773.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 1764.10 | 1754.46 | 1758.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1764.60 | 1756.49 | 1759.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1789.60 | 1773.91 | 1768.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 1794.50 | 1801.01 | 1791.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1794.50 | 1801.01 | 1791.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1794.50 | 1801.01 | 1791.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1794.50 | 1801.01 | 1791.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1795.80 | 1799.97 | 1792.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:15:00 | 1793.10 | 1799.97 | 1792.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1794.30 | 1798.84 | 1792.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:15:00 | 1792.50 | 1798.84 | 1792.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1784.10 | 1795.89 | 1791.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1784.10 | 1795.89 | 1791.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1787.00 | 1794.11 | 1791.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1788.90 | 1794.11 | 1791.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1786.90 | 1792.67 | 1790.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1793.10 | 1791.76 | 1790.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1777.20 | 1788.84 | 1789.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 1777.20 | 1788.84 | 1789.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1770.20 | 1782.20 | 1785.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1727.10 | 1725.60 | 1735.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:15:00 | 1726.10 | 1725.60 | 1735.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1733.40 | 1728.13 | 1735.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 1734.40 | 1728.13 | 1735.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1722.10 | 1725.21 | 1730.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 1722.10 | 1725.21 | 1730.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1725.20 | 1724.20 | 1728.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:30:00 | 1728.10 | 1724.20 | 1728.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1724.10 | 1724.18 | 1728.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 1742.60 | 1724.18 | 1728.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1743.70 | 1728.08 | 1729.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 1751.20 | 1728.08 | 1729.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 10:15:00 | 1742.70 | 1731.01 | 1730.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 1772.70 | 1743.50 | 1736.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 12:15:00 | 1755.60 | 1760.44 | 1749.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 1755.60 | 1760.44 | 1749.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1741.80 | 1756.71 | 1748.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:30:00 | 1742.00 | 1756.71 | 1748.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1737.00 | 1752.77 | 1747.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1737.00 | 1752.77 | 1747.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 1724.20 | 1744.05 | 1744.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 11:15:00 | 1718.70 | 1738.98 | 1742.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 1737.90 | 1735.42 | 1739.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 1776.00 | 1735.42 | 1739.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 187 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1770.60 | 1742.45 | 1741.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 1804.40 | 1763.25 | 1753.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 13:15:00 | 1791.20 | 1797.85 | 1776.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:30:00 | 1795.00 | 1797.85 | 1776.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1790.00 | 1792.49 | 1780.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 1787.50 | 1792.49 | 1780.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1787.90 | 1791.58 | 1780.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1784.20 | 1791.58 | 1780.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1783.90 | 1790.04 | 1781.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 1785.00 | 1790.04 | 1781.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1769.20 | 1785.87 | 1780.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1769.20 | 1785.87 | 1780.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1781.00 | 1784.90 | 1780.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1783.60 | 1784.32 | 1780.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 1771.70 | 1777.87 | 1777.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 1771.70 | 1777.87 | 1777.97 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 1784.20 | 1779.13 | 1778.54 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1777.50 | 1778.08 | 1778.12 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 1785.00 | 1779.46 | 1778.74 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 1754.00 | 1773.58 | 1776.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1740.70 | 1756.75 | 1763.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1753.20 | 1749.53 | 1756.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1753.20 | 1749.53 | 1756.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1753.20 | 1749.53 | 1756.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1758.70 | 1749.53 | 1756.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1765.80 | 1752.78 | 1757.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1765.80 | 1752.78 | 1757.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1755.40 | 1753.30 | 1757.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1752.00 | 1753.30 | 1757.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1768.40 | 1754.49 | 1755.90 | SL hit (close>static) qty=1.00 sl=1767.90 alert=retest2 |

### Cycle 193 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1764.10 | 1757.67 | 1757.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 1782.00 | 1764.48 | 1760.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1759.20 | 1763.75 | 1761.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 1759.20 | 1763.75 | 1761.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1759.20 | 1763.75 | 1761.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1759.00 | 1763.75 | 1761.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1755.80 | 1762.16 | 1760.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 1755.80 | 1762.16 | 1760.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1757.00 | 1761.13 | 1760.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1766.00 | 1761.13 | 1760.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1756.00 | 1762.71 | 1763.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 1756.00 | 1762.71 | 1763.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1743.90 | 1757.00 | 1760.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 1739.10 | 1737.44 | 1744.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1739.10 | 1737.44 | 1744.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1739.10 | 1737.44 | 1744.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1739.10 | 1737.44 | 1744.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1747.10 | 1739.37 | 1744.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1747.10 | 1739.37 | 1744.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1750.10 | 1741.52 | 1745.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1763.90 | 1741.52 | 1745.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1776.00 | 1752.51 | 1749.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1780.70 | 1770.94 | 1762.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 1784.10 | 1789.38 | 1778.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 11:15:00 | 1784.10 | 1789.38 | 1778.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1784.10 | 1789.38 | 1778.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 1782.00 | 1789.38 | 1778.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1768.00 | 1786.42 | 1781.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1768.00 | 1786.42 | 1781.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1747.00 | 1778.53 | 1778.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1747.00 | 1778.53 | 1778.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1745.30 | 1771.89 | 1775.37 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1786.90 | 1769.99 | 1768.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 1835.70 | 1785.19 | 1776.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1821.60 | 1821.82 | 1803.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:30:00 | 1826.40 | 1821.82 | 1803.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1808.10 | 1818.79 | 1811.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:00:00 | 1824.10 | 1813.60 | 1810.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1781.10 | 1804.35 | 1807.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1781.10 | 1804.35 | 1807.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 1740.80 | 1768.41 | 1785.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 14:15:00 | 1758.70 | 1755.84 | 1771.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 15:00:00 | 1758.70 | 1755.84 | 1771.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1773.80 | 1759.99 | 1770.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1773.80 | 1759.99 | 1770.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1774.80 | 1762.95 | 1771.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1774.80 | 1762.95 | 1771.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1794.80 | 1769.32 | 1773.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 1794.80 | 1769.32 | 1773.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1765.30 | 1772.76 | 1774.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 1757.10 | 1768.31 | 1771.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 1756.30 | 1765.90 | 1770.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1921.90 | 1792.22 | 1780.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1921.90 | 1792.22 | 1780.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 1942.00 | 1854.70 | 1814.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 12:15:00 | 1937.40 | 1942.45 | 1890.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:00:00 | 1937.40 | 1942.45 | 1890.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2026.30 | 2010.06 | 1984.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 2035.40 | 2017.95 | 1992.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 2052.00 | 2024.76 | 1997.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1981.80 | 2006.36 | 2003.88 | SL hit (close<static) qty=1.00 sl=1982.70 alert=retest2 |

### Cycle 200 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1982.20 | 2001.53 | 2001.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 1948.90 | 1987.24 | 1995.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1970.30 | 1966.62 | 1978.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:30:00 | 1971.90 | 1966.62 | 1978.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1978.30 | 1968.95 | 1978.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1980.80 | 1968.95 | 1978.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1968.90 | 1968.94 | 1977.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:00:00 | 1959.90 | 1966.04 | 1974.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 1952.00 | 1966.21 | 1971.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 1959.80 | 1949.91 | 1956.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:00:00 | 1948.80 | 1926.66 | 1928.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 1914.60 | 1934.02 | 1936.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1893.80 | 1919.57 | 1928.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 1923.40 | 1918.82 | 1926.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 1923.40 | 1918.82 | 1926.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1923.40 | 1918.82 | 1926.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 1923.40 | 1918.82 | 1926.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 1922.30 | 1919.52 | 1925.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 1920.80 | 1919.52 | 1925.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1918.00 | 1919.21 | 1925.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 1891.50 | 1914.49 | 1922.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 1901.00 | 1885.69 | 1884.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 1901.00 | 1885.69 | 1884.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 1905.00 | 1892.02 | 1887.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1886.40 | 1890.89 | 1887.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1886.40 | 1890.89 | 1887.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1886.40 | 1890.89 | 1887.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1893.20 | 1890.89 | 1887.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1913.00 | 1895.32 | 1889.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 1919.10 | 1895.32 | 1889.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1872.00 | 1892.71 | 1891.95 | SL hit (close<static) qty=1.00 sl=1880.80 alert=retest2 |

### Cycle 204 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1872.20 | 1888.61 | 1890.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1853.00 | 1874.29 | 1881.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1873.80 | 1856.57 | 1866.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1873.80 | 1856.57 | 1866.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1873.80 | 1856.57 | 1866.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1873.80 | 1856.57 | 1866.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1878.20 | 1860.90 | 1867.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 1882.90 | 1860.90 | 1867.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1865.40 | 1863.73 | 1868.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1878.30 | 1863.73 | 1868.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1874.80 | 1865.95 | 1868.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1874.80 | 1865.95 | 1868.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1888.90 | 1870.54 | 1870.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 1920.00 | 1880.43 | 1874.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1883.40 | 1892.90 | 1886.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1883.40 | 1892.90 | 1886.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1883.40 | 1892.90 | 1886.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1876.10 | 1892.90 | 1886.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1880.00 | 1890.32 | 1885.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 1882.60 | 1890.32 | 1885.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 1870.20 | 1883.77 | 1883.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1867.70 | 1875.58 | 1878.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 1877.00 | 1875.86 | 1878.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 1877.00 | 1875.86 | 1878.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1877.00 | 1875.86 | 1878.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1877.00 | 1875.86 | 1878.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1877.00 | 1876.09 | 1878.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 1877.00 | 1876.09 | 1878.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1869.40 | 1874.75 | 1877.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:15:00 | 1865.00 | 1873.56 | 1876.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:45:00 | 1866.00 | 1868.04 | 1873.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 1863.30 | 1865.95 | 1871.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:45:00 | 1833.50 | 1835.40 | 1849.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1851.20 | 1839.30 | 1848.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 1851.20 | 1839.30 | 1848.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | SL hit (close>static) qty=1.00 sl=1881.00 alert=retest2 |

### Cycle 207 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1867.20 | 1856.38 | 1855.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 1890.00 | 1863.10 | 1858.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 1916.90 | 1920.71 | 1905.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 15:00:00 | 1916.90 | 1920.71 | 1905.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1892.80 | 1915.37 | 1905.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1892.80 | 1915.37 | 1905.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1923.60 | 1917.01 | 1907.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1931.00 | 1917.01 | 1907.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1929.40 | 1906.62 | 1905.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 1930.70 | 1912.91 | 1908.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 1937.10 | 1917.75 | 1911.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1922.70 | 1926.05 | 1919.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 1916.00 | 1926.05 | 1919.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1924.10 | 1925.66 | 1919.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 1923.80 | 1925.66 | 1919.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1923.10 | 1924.63 | 1920.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1923.10 | 1924.63 | 1920.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1917.50 | 1923.20 | 1920.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1925.80 | 1923.20 | 1920.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1935.90 | 1925.74 | 1921.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1894.90 | 1913.74 | 1916.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 15:15:00 | 1920.00 | 1914.23 | 1916.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 15:15:00 | 1920.00 | 1914.23 | 1916.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1920.00 | 1914.23 | 1916.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1899.30 | 1914.23 | 1916.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1896.40 | 1910.88 | 1914.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1937.90 | 1897.24 | 1891.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1937.90 | 1897.24 | 1891.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1944.20 | 1912.61 | 1900.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1922.60 | 1926.66 | 1912.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1916.80 | 1922.40 | 1914.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1916.80 | 1922.40 | 1914.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 1914.10 | 1922.40 | 1914.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1916.00 | 1921.12 | 1914.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:15:00 | 1915.00 | 1921.12 | 1914.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1915.00 | 1919.89 | 1914.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1902.60 | 1919.89 | 1914.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1900.90 | 1916.10 | 1913.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 1898.60 | 1916.10 | 1913.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1885.00 | 1909.88 | 1910.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1867.40 | 1901.38 | 1906.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1817.20 | 1801.49 | 1827.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1817.20 | 1801.49 | 1827.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1803.40 | 1804.84 | 1824.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1794.40 | 1802.54 | 1819.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 1798.70 | 1792.05 | 1803.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1708.76 | 1765.89 | 1786.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1769.70 | 1741.32 | 1759.23 | SL hit (close>ema200) qty=0.50 sl=1741.32 alert=retest2 |

### Cycle 211 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 1766.80 | 1757.09 | 1755.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 1789.90 | 1768.51 | 1762.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1741.70 | 1777.61 | 1771.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1741.70 | 1777.61 | 1771.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1741.70 | 1777.61 | 1771.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 1741.70 | 1777.61 | 1771.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1728.00 | 1767.69 | 1767.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1728.00 | 1767.69 | 1767.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1738.20 | 1761.79 | 1764.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1694.00 | 1728.38 | 1740.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1716.00 | 1714.41 | 1728.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 1716.00 | 1714.41 | 1728.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1735.20 | 1719.08 | 1727.32 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1742.40 | 1732.87 | 1732.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1788.90 | 1749.10 | 1740.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 13:15:00 | 1862.20 | 1862.34 | 1833.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:45:00 | 1866.20 | 1862.34 | 1833.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1843.00 | 1859.52 | 1839.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 1836.10 | 1859.52 | 1839.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1845.40 | 1856.70 | 1840.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 1862.00 | 1857.76 | 1842.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1828.80 | 1849.93 | 1841.16 | SL hit (close<static) qty=1.00 sl=1831.10 alert=retest2 |

### Cycle 214 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 1815.40 | 1834.68 | 1835.63 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1839.10 | 1834.96 | 1834.72 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 14:15:00 | 1828.60 | 1833.53 | 1834.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 1824.10 | 1830.44 | 1832.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 1833.20 | 1830.99 | 1832.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 10:15:00 | 1833.20 | 1830.99 | 1832.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1833.20 | 1830.99 | 1832.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 1833.20 | 1830.99 | 1832.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 1851.50 | 1835.09 | 1834.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1852.90 | 1844.09 | 1839.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1836.10 | 1844.35 | 1841.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1836.10 | 1844.35 | 1841.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1836.10 | 1844.35 | 1841.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1836.10 | 1844.35 | 1841.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1845.00 | 1844.48 | 1841.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1817.90 | 1844.48 | 1841.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1818.00 | 1839.18 | 1839.63 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1873.20 | 1840.92 | 1839.08 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1817.90 | 1842.13 | 1844.11 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1847.60 | 1839.86 | 1839.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1864.50 | 1844.79 | 1841.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1853.10 | 1853.43 | 1847.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 1852.70 | 1853.43 | 1847.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1850.20 | 1853.16 | 1848.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:30:00 | 1851.10 | 1853.16 | 1848.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1853.60 | 1853.83 | 1849.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 1837.60 | 1853.83 | 1849.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1837.40 | 1850.55 | 1848.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1839.40 | 1850.55 | 1848.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 1825.00 | 1845.44 | 1846.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1821.20 | 1840.59 | 1844.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 15:15:00 | 1853.90 | 1839.02 | 1842.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 1853.90 | 1839.02 | 1842.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1853.90 | 1839.02 | 1842.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1797.90 | 1839.02 | 1842.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1708.01 | 1747.36 | 1755.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 1756.90 | 1747.45 | 1754.34 | SL hit (close>ema200) qty=0.50 sl=1747.45 alert=retest2 |

### Cycle 223 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1773.80 | 1757.96 | 1757.24 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 1749.30 | 1756.29 | 1756.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 1745.00 | 1751.82 | 1754.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 15:15:00 | 1645.00 | 1644.25 | 1668.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:15:00 | 1650.40 | 1644.25 | 1668.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1653.80 | 1649.81 | 1661.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 1663.70 | 1649.81 | 1661.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1665.10 | 1652.87 | 1661.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1681.00 | 1652.87 | 1661.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1695.00 | 1661.29 | 1664.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1697.90 | 1661.29 | 1664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1688.10 | 1666.66 | 1666.65 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1648.80 | 1671.64 | 1672.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1612.00 | 1652.15 | 1660.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1642.40 | 1632.89 | 1643.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1642.40 | 1632.89 | 1643.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1642.40 | 1632.89 | 1643.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 1644.50 | 1632.89 | 1643.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1640.80 | 1634.47 | 1643.21 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 1666.50 | 1648.86 | 1648.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1672.50 | 1655.27 | 1651.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1678.70 | 1690.29 | 1675.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1678.70 | 1690.29 | 1675.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1678.70 | 1690.29 | 1675.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1678.70 | 1690.29 | 1675.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1655.70 | 1683.37 | 1673.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1655.70 | 1683.37 | 1673.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1627.10 | 1672.12 | 1669.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1627.10 | 1672.12 | 1669.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1622.10 | 1662.11 | 1664.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1609.50 | 1636.82 | 1650.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1630.80 | 1622.15 | 1635.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1630.80 | 1622.15 | 1635.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1630.80 | 1622.15 | 1635.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1618.60 | 1622.15 | 1635.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 1627.00 | 1626.80 | 1633.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1596.90 | 1630.20 | 1634.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1612.90 | 1618.49 | 1623.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1635.40 | 1622.10 | 1624.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 1635.40 | 1622.10 | 1624.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1670.20 | 1640.88 | 1633.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1721.00 | 1724.22 | 1710.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1743.40 | 1724.22 | 1710.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1748.20 | 1747.89 | 1732.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1754.30 | 1747.89 | 1732.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1772.20 | 1789.62 | 1777.38 | SL hit (close<ema400) qty=1.00 sl=1777.38 alert=retest1 |

### Cycle 230 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1793.20 | 1804.73 | 1806.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1782.20 | 1797.02 | 1801.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1802.00 | 1784.85 | 1791.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1802.00 | 1784.85 | 1791.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1802.00 | 1784.85 | 1791.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1806.20 | 1784.85 | 1791.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1803.90 | 1788.66 | 1792.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1804.60 | 1788.66 | 1792.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1803.10 | 1795.46 | 1795.20 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 1788.20 | 1794.01 | 1794.56 | EMA200 below EMA400 |

### Cycle 233 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1799.00 | 1795.01 | 1794.96 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1786.40 | 1793.29 | 1794.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 10:15:00 | 1783.70 | 1791.37 | 1793.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 1776.80 | 1772.73 | 1779.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 12:00:00 | 1776.80 | 1772.73 | 1779.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1775.10 | 1773.21 | 1779.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1771.70 | 1773.25 | 1778.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 1769.00 | 1773.25 | 1778.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:45:00 | 1768.80 | 1769.92 | 1775.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:00:00 | 1770.50 | 1766.28 | 1771.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1768.30 | 1766.68 | 1771.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1768.70 | 1766.68 | 1771.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1773.00 | 1767.95 | 1771.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1783.70 | 1767.95 | 1771.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | EMA200 above EMA400 |

### Cycle 236 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 1762.30 | 1777.42 | 1779.05 | EMA200 below EMA400 |

### Cycle 237 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1802.60 | 1782.24 | 1780.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1822.90 | 1790.37 | 1784.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 1832.10 | 1848.70 | 1831.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 1832.10 | 1848.70 | 1831.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1832.10 | 1848.70 | 1831.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1832.10 | 1848.70 | 1831.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1823.80 | 1843.72 | 1830.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 1824.30 | 1843.72 | 1830.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1825.00 | 1839.98 | 1830.05 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-16 12:15:00 | 1281.00 | 2024-04-19 09:15:00 | 1216.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-16 12:45:00 | 1279.45 | 2024-04-19 09:15:00 | 1215.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-16 12:15:00 | 1281.00 | 2024-04-19 11:15:00 | 1247.80 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2024-04-16 12:45:00 | 1279.45 | 2024-04-19 11:15:00 | 1247.80 | STOP_HIT | 0.50 | 2.47% |
| BUY | retest2 | 2024-04-26 09:30:00 | 1275.85 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-04-29 10:15:00 | 1271.60 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-04-29 13:00:00 | 1272.60 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-04-29 15:15:00 | 1271.00 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-04-30 09:15:00 | 1287.30 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-05-02 10:00:00 | 1271.20 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-05-02 11:30:00 | 1271.90 | 2024-05-02 12:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-05-14 09:30:00 | 1254.00 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-05-14 10:30:00 | 1254.30 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-05-14 14:00:00 | 1253.00 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-05-15 09:30:00 | 1253.30 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-05-15 13:45:00 | 1253.55 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-05-16 10:00:00 | 1255.10 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-05-16 13:15:00 | 1251.00 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-05-16 15:00:00 | 1255.10 | 2024-05-17 09:15:00 | 1276.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-05-28 09:15:00 | 1205.70 | 2024-05-30 09:15:00 | 1145.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:15:00 | 1205.70 | 2024-05-30 11:15:00 | 1151.20 | STOP_HIT | 0.50 | 4.52% |
| BUY | retest2 | 2024-06-11 11:15:00 | 1277.20 | 2024-06-11 14:15:00 | 1254.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-06-14 13:45:00 | 1238.65 | 2024-06-18 09:15:00 | 1253.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-06-20 11:15:00 | 1232.00 | 2024-06-27 09:15:00 | 1231.70 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-06-20 13:15:00 | 1233.85 | 2024-06-27 09:15:00 | 1231.70 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-06-20 14:00:00 | 1233.00 | 2024-06-27 09:15:00 | 1231.70 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-07-24 09:15:00 | 1268.00 | 2024-07-25 09:15:00 | 1240.10 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-07-24 14:30:00 | 1258.00 | 2024-07-25 09:15:00 | 1240.10 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-08-06 10:30:00 | 1235.95 | 2024-08-09 09:15:00 | 1238.15 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-08-06 11:00:00 | 1236.90 | 2024-08-09 09:15:00 | 1238.15 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2024-08-14 09:30:00 | 1194.10 | 2024-08-19 14:15:00 | 1187.75 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest1 | 2024-08-14 10:30:00 | 1194.75 | 2024-08-19 14:15:00 | 1187.75 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest1 | 2024-08-14 11:30:00 | 1192.00 | 2024-08-19 14:15:00 | 1187.75 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest1 | 2024-08-14 12:45:00 | 1195.90 | 2024-08-19 14:15:00 | 1187.75 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2024-08-28 09:15:00 | 1282.00 | 2024-08-29 13:15:00 | 1266.85 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-29 12:15:00 | 1278.00 | 2024-08-29 13:15:00 | 1266.85 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-09-11 09:15:00 | 1358.00 | 2024-09-11 09:15:00 | 1341.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-09-11 10:15:00 | 1356.00 | 2024-09-11 11:15:00 | 1342.55 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-11 10:45:00 | 1352.85 | 2024-09-11 11:15:00 | 1342.55 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-20 13:45:00 | 1245.00 | 2024-09-25 11:15:00 | 1259.95 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-20 14:15:00 | 1240.00 | 2024-09-25 11:15:00 | 1259.95 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-09-23 09:15:00 | 1245.50 | 2024-09-25 11:15:00 | 1259.95 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1210.85 | 2024-10-08 10:15:00 | 1232.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-10-07 15:15:00 | 1222.00 | 2024-10-08 10:15:00 | 1232.40 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-10-10 09:30:00 | 1228.10 | 2024-10-11 13:15:00 | 1221.20 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-10-10 14:15:00 | 1226.25 | 2024-10-11 13:15:00 | 1221.20 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-10-10 15:15:00 | 1226.30 | 2024-10-11 13:15:00 | 1221.20 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-10-11 09:30:00 | 1229.15 | 2024-10-11 13:15:00 | 1221.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-10-28 12:45:00 | 1240.95 | 2024-10-31 10:15:00 | 1251.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-10-29 09:30:00 | 1230.00 | 2024-10-31 10:15:00 | 1251.45 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-26 12:30:00 | 1259.05 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-11-26 13:00:00 | 1259.00 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-11-26 13:30:00 | 1254.35 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-11-26 14:45:00 | 1257.50 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-11-27 09:15:00 | 1258.55 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-11-27 11:00:00 | 1259.35 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-11-28 10:30:00 | 1256.25 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-11-29 10:00:00 | 1255.05 | 2024-11-29 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-12-06 09:30:00 | 1318.80 | 2024-12-10 13:15:00 | 1302.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-12-10 10:15:00 | 1317.85 | 2024-12-10 13:15:00 | 1302.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-12-12 12:45:00 | 1291.90 | 2024-12-16 11:15:00 | 1298.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-12 15:15:00 | 1292.00 | 2024-12-16 11:15:00 | 1298.90 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-12-16 09:30:00 | 1290.45 | 2024-12-16 11:15:00 | 1298.90 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-12-26 10:00:00 | 1314.90 | 2024-12-27 14:15:00 | 1297.10 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-12-27 11:30:00 | 1309.00 | 2024-12-27 14:15:00 | 1297.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-27 12:15:00 | 1308.95 | 2024-12-27 14:15:00 | 1297.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-27 13:00:00 | 1307.80 | 2024-12-27 14:15:00 | 1297.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-03 09:15:00 | 1302.55 | 2025-01-09 14:15:00 | 1328.60 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-01-22 14:45:00 | 1322.00 | 2025-01-24 13:15:00 | 1298.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-01-23 09:30:00 | 1327.70 | 2025-01-24 13:15:00 | 1298.90 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-01-23 12:15:00 | 1321.65 | 2025-01-24 13:15:00 | 1298.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-01-24 10:30:00 | 1317.60 | 2025-01-24 13:15:00 | 1298.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-02-06 10:00:00 | 1380.65 | 2025-02-11 09:15:00 | 1358.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-02-06 14:30:00 | 1379.00 | 2025-02-11 10:15:00 | 1349.95 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-02-07 09:15:00 | 1380.50 | 2025-02-11 10:15:00 | 1349.95 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-02-07 10:30:00 | 1395.45 | 2025-02-11 10:15:00 | 1349.95 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-02-10 09:15:00 | 1407.20 | 2025-02-11 10:15:00 | 1349.95 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest1 | 2025-02-12 09:15:00 | 1336.00 | 2025-02-12 12:15:00 | 1367.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-02-13 11:30:00 | 1353.00 | 2025-02-18 10:15:00 | 1364.45 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1396.10 | 2025-03-03 12:15:00 | 1535.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-24 11:30:00 | 1685.00 | 2025-03-25 11:15:00 | 1645.90 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-04-02 14:15:00 | 1686.00 | 2025-04-03 10:15:00 | 1652.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-04-11 14:00:00 | 1705.05 | 2025-04-23 12:15:00 | 1798.80 | STOP_HIT | 1.00 | 5.50% |
| BUY | retest2 | 2025-04-15 09:15:00 | 1726.50 | 2025-04-23 12:15:00 | 1798.80 | STOP_HIT | 1.00 | 4.19% |
| SELL | retest2 | 2025-05-13 11:15:00 | 1771.10 | 2025-05-14 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-05-21 11:45:00 | 1744.40 | 2025-05-26 09:15:00 | 1822.30 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-06-13 12:45:00 | 1886.50 | 2025-06-19 12:15:00 | 1878.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-06-20 14:00:00 | 1863.00 | 2025-06-20 14:15:00 | 1903.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-06-27 11:15:00 | 2045.90 | 2025-06-27 14:15:00 | 2250.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 12:45:00 | 2044.00 | 2025-06-27 14:15:00 | 2248.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-03 12:15:00 | 2044.20 | 2025-07-11 10:15:00 | 2009.90 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1949.20 | 2025-07-22 09:15:00 | 1971.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-18 13:15:00 | 1940.70 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-07-18 14:15:00 | 1934.60 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-21 09:15:00 | 1935.30 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1929.90 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-07-28 09:15:00 | 1999.60 | 2025-07-29 09:15:00 | 1972.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1943.00 | 2025-08-04 09:15:00 | 1845.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 11:45:00 | 1944.50 | 2025-08-04 09:15:00 | 1847.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 12:30:00 | 1941.00 | 2025-08-04 09:15:00 | 1843.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:30:00 | 1935.70 | 2025-08-04 09:15:00 | 1838.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 09:15:00 | 1861.30 | 2025-08-06 09:15:00 | 1768.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1943.00 | 2025-08-07 09:15:00 | 1748.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 11:45:00 | 1944.50 | 2025-08-07 09:15:00 | 1750.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 12:30:00 | 1941.00 | 2025-08-07 09:15:00 | 1746.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 13:30:00 | 1935.70 | 2025-08-07 09:15:00 | 1742.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-04 09:15:00 | 1861.30 | 2025-08-07 12:15:00 | 1780.30 | STOP_HIT | 0.50 | 4.35% |
| BUY | retest2 | 2025-08-18 09:30:00 | 1831.60 | 2025-08-22 10:15:00 | 1823.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-01 15:15:00 | 1782.90 | 2025-09-01 15:15:00 | 1782.90 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-09 11:30:00 | 1763.40 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-11 09:30:00 | 1763.50 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-11 10:00:00 | 1754.10 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-09-12 13:00:00 | 1764.10 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1793.10 | 2025-09-19 09:15:00 | 1777.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1783.60 | 2025-10-08 11:15:00 | 1771.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-15 12:15:00 | 1752.00 | 2025-10-16 09:15:00 | 1768.40 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-16 11:30:00 | 1754.10 | 2025-10-16 13:15:00 | 1764.10 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-16 12:30:00 | 1755.20 | 2025-10-16 13:15:00 | 1764.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-20 09:15:00 | 1766.00 | 2025-10-23 11:15:00 | 1756.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-10 10:00:00 | 1824.10 | 2025-11-11 09:15:00 | 1781.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-11-14 11:45:00 | 1757.10 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -9.38% |
| SELL | retest2 | 2025-11-14 13:00:00 | 1756.30 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -9.43% |
| BUY | retest2 | 2025-11-21 12:00:00 | 2035.40 | 2025-11-24 13:15:00 | 1981.80 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-11-21 13:00:00 | 2052.00 | 2025-11-24 13:15:00 | 1981.80 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-11-26 14:00:00 | 1959.90 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-11-27 13:30:00 | 1952.00 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-12-01 09:30:00 | 1959.80 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-12-03 13:00:00 | 1948.80 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-09 10:45:00 | 1891.50 | 2025-12-15 13:15:00 | 1901.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-16 11:15:00 | 1919.10 | 2025-12-17 09:15:00 | 1872.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-26 14:15:00 | 1865.00 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 09:45:00 | 1866.00 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-29 10:30:00 | 1863.30 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-30 14:45:00 | 1833.50 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-01-05 11:15:00 | 1931.00 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1929.40 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-06 12:00:00 | 1930.70 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-06 13:00:00 | 1937.10 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-09 09:15:00 | 1899.30 | 2026-01-14 10:15:00 | 1937.90 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-01-09 13:45:00 | 1896.40 | 2026-01-14 10:15:00 | 1937.90 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1794.40 | 2026-01-27 09:15:00 | 1708.76 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1794.40 | 2026-01-28 09:15:00 | 1769.70 | STOP_HIT | 0.50 | 1.38% |
| SELL | retest2 | 2026-01-23 11:30:00 | 1798.70 | 2026-01-30 12:15:00 | 1766.80 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-02-13 12:00:00 | 1862.00 | 2026-02-13 13:15:00 | 1828.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-09 09:15:00 | 1708.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-09 11:15:00 | 1756.90 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1618.60 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-01 13:30:00 | 1627.00 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1596.90 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-06 09:15:00 | 1612.90 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1743.40 | 2026-04-16 11:15:00 | 1772.20 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1754.30 | 2026-04-23 11:15:00 | 1793.20 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1771.70 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-29 15:15:00 | 1769.00 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-04-30 09:45:00 | 1768.80 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-04-30 14:00:00 | 1770.50 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.91% |
