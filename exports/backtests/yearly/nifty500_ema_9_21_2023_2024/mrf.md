# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 130490.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 201 |
| ALERT1 | 138 |
| ALERT2 | 137 |
| ALERT2_SKIP | 89 |
| ALERT3 | 270 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 128 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 133 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 33 / 107
- **Target hits / Stop hits / Partials:** 0 / 133 / 7
- **Avg / median % per leg:** -0.03% / -0.55%
- **Sum % (uncompounded):** -4.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 11 | 19.3% | 0 | 57 | 0 | -0.46% | -26.3% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.19% | -7.1% |
| BUY @ 3rd Alert (retest2) | 51 | 11 | 21.6% | 0 | 51 | 0 | -0.38% | -19.1% |
| SELL (all) | 83 | 22 | 26.5% | 0 | 76 | 7 | 0.27% | 22.1% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.70% | -1.4% |
| SELL @ 3rd Alert (retest2) | 81 | 22 | 27.2% | 0 | 74 | 7 | 0.29% | 23.5% |
| retest1 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.06% | -8.5% |
| retest2 (combined) | 132 | 33 | 25.0% | 0 | 125 | 7 | 0.03% | 4.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 11:15:00 | 97407.80 | 96952.02 | 96941.31 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 96560.80 | 96889.88 | 96918.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 15:15:00 | 96420.00 | 96795.91 | 96873.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 97306.60 | 96898.04 | 96912.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 09:15:00 | 97306.60 | 96898.04 | 96912.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 97306.60 | 96898.04 | 96912.82 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 10:15:00 | 97336.80 | 96985.80 | 96951.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 14:15:00 | 97600.00 | 97156.56 | 97046.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 11:15:00 | 97161.00 | 97321.24 | 97178.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 11:15:00 | 97161.00 | 97321.24 | 97178.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 97161.00 | 97321.24 | 97178.20 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 95789.90 | 96907.74 | 97026.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 15:15:00 | 95674.20 | 96042.92 | 96358.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 96234.00 | 96081.14 | 96347.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 11:15:00 | 96055.00 | 96100.90 | 96311.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 96055.00 | 96100.90 | 96311.68 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 96810.00 | 96275.79 | 96227.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 97500.10 | 96590.29 | 96382.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 97511.60 | 97647.62 | 97356.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 12:15:00 | 97378.50 | 97560.58 | 97365.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 97378.50 | 97560.58 | 97365.68 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 10:15:00 | 96989.00 | 97460.67 | 97473.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 11:15:00 | 96679.10 | 97304.36 | 97400.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 09:15:00 | 96950.00 | 96856.42 | 97109.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 09:15:00 | 96950.00 | 96856.42 | 97109.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 96950.00 | 96856.42 | 97109.71 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 97950.00 | 96867.11 | 96857.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 10:15:00 | 98030.00 | 97516.59 | 97371.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 15:15:00 | 99800.00 | 99891.49 | 99466.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 14:15:00 | 100111.00 | 100053.06 | 99757.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 100111.00 | 100053.06 | 99757.97 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 15:15:00 | 99742.60 | 99959.79 | 99967.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 09:15:00 | 99465.10 | 99860.85 | 99921.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 15:15:00 | 99472.90 | 99332.29 | 99560.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 15:15:00 | 99472.90 | 99332.29 | 99560.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 99472.90 | 99332.29 | 99560.20 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 11:15:00 | 100600.00 | 99799.89 | 99736.89 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 99357.30 | 99879.49 | 99919.76 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 10:15:00 | 99795.10 | 99738.90 | 99738.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 11:15:00 | 99894.00 | 99769.92 | 99752.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 14:15:00 | 99601.80 | 99779.31 | 99765.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 14:15:00 | 99601.80 | 99779.31 | 99765.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 99601.80 | 99779.31 | 99765.71 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 99500.00 | 99723.45 | 99741.56 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 99940.90 | 99758.23 | 99753.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 12:15:00 | 99978.10 | 99824.89 | 99786.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 100420.00 | 100722.35 | 100403.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 100420.00 | 100722.35 | 100403.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 100420.00 | 100722.35 | 100403.58 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 13:15:00 | 99754.00 | 100268.74 | 100273.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 09:15:00 | 99569.50 | 100045.15 | 100161.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 99495.60 | 99479.17 | 99752.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 11:15:00 | 99855.00 | 99562.63 | 99743.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 11:15:00 | 99855.00 | 99562.63 | 99743.39 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 15:15:00 | 100296.00 | 99894.94 | 99856.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 100603.00 | 100036.55 | 99924.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 15:15:00 | 100920.00 | 101037.91 | 100750.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 11:15:00 | 100500.00 | 100988.37 | 100805.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 100500.00 | 100988.37 | 100805.33 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 09:15:00 | 101535.00 | 101626.93 | 101628.01 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 10:15:00 | 101936.00 | 101688.74 | 101656.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 15:15:00 | 102300.00 | 101927.91 | 101796.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 10:15:00 | 101939.00 | 101977.66 | 101844.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 12:15:00 | 101960.00 | 101973.71 | 101865.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 101960.00 | 101973.71 | 101865.92 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 14:15:00 | 101890.00 | 102135.17 | 102143.82 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 102515.00 | 102057.11 | 102045.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 12:15:00 | 102840.00 | 102262.15 | 102143.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 102400.00 | 102434.60 | 102264.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 102539.00 | 102455.48 | 102289.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 102539.00 | 102455.48 | 102289.21 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 09:15:00 | 102219.00 | 102425.19 | 102426.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 11:15:00 | 102121.00 | 102345.44 | 102388.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 102730.00 | 102224.08 | 102264.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 14:15:00 | 102730.00 | 102224.08 | 102264.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 102730.00 | 102224.08 | 102264.42 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 102500.00 | 102326.73 | 102306.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 12:15:00 | 102644.00 | 102395.99 | 102341.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 102810.00 | 103018.76 | 102823.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 102810.00 | 103018.76 | 102823.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 102810.00 | 103018.76 | 102823.57 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 102575.00 | 102741.39 | 102743.42 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 12:15:00 | 105120.00 | 103034.23 | 102854.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 14:15:00 | 106750.00 | 104179.91 | 103431.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 11:15:00 | 108755.00 | 109580.78 | 107730.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 14:15:00 | 108308.00 | 109026.13 | 107913.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 108308.00 | 109026.13 | 107913.22 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 106166.00 | 107278.46 | 107399.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 10:15:00 | 105811.00 | 106380.78 | 106779.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 14:15:00 | 106375.00 | 106280.05 | 106594.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 15:15:00 | 106600.00 | 106344.04 | 106595.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 106600.00 | 106344.04 | 106595.08 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 14:15:00 | 106276.00 | 106118.17 | 106114.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 13:15:00 | 106674.00 | 106426.38 | 106295.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 15:15:00 | 108400.00 | 108568.13 | 108239.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 15:15:00 | 108400.00 | 108568.13 | 108239.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 108400.00 | 108568.13 | 108239.84 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 108120.00 | 108879.46 | 108946.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 10:15:00 | 107763.00 | 108656.17 | 108838.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 108784.00 | 108313.59 | 108529.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 108784.00 | 108313.59 | 108529.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 108784.00 | 108313.59 | 108529.70 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 12:15:00 | 109100.00 | 108390.28 | 108344.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 109391.00 | 108712.30 | 108506.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 15:15:00 | 108801.00 | 109107.36 | 108891.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 15:15:00 | 108801.00 | 109107.36 | 108891.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 108801.00 | 109107.36 | 108891.06 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 107832.00 | 108857.84 | 108977.93 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 109551.00 | 108819.76 | 108737.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 13:15:00 | 110740.00 | 109721.87 | 109234.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 11:15:00 | 110285.00 | 110574.18 | 109913.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 12:15:00 | 110200.00 | 110499.35 | 109939.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 12:15:00 | 110200.00 | 110499.35 | 109939.33 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 109000.00 | 109583.16 | 109642.35 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 15:15:00 | 109700.00 | 109415.92 | 109407.83 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 109098.00 | 109352.34 | 109379.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 108830.00 | 109196.38 | 109299.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 11:15:00 | 109200.00 | 108973.73 | 109115.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 11:15:00 | 109200.00 | 108973.73 | 109115.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 109200.00 | 108973.73 | 109115.47 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 109350.00 | 108973.46 | 108926.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 11:15:00 | 109574.00 | 109093.57 | 108985.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 109500.00 | 109500.70 | 109261.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 09:15:00 | 109500.00 | 109500.70 | 109261.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 109500.00 | 109500.70 | 109261.05 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 108634.00 | 109372.42 | 109424.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 11:15:00 | 108300.00 | 108990.14 | 109215.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 14:15:00 | 107984.00 | 107780.67 | 108215.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 15:15:00 | 108389.00 | 107902.34 | 108231.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 108389.00 | 107902.34 | 108231.35 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 107600.00 | 107493.83 | 107480.91 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 106806.00 | 107356.26 | 107419.56 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 107917.00 | 107423.13 | 107416.14 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 14:15:00 | 107169.00 | 107413.85 | 107426.58 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 108041.00 | 107520.90 | 107471.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 108318.00 | 107814.14 | 107661.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 15:15:00 | 110700.00 | 110850.29 | 110153.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 110364.00 | 110753.03 | 110173.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 110364.00 | 110753.03 | 110173.08 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 109099.00 | 109875.76 | 109962.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 108205.00 | 109119.58 | 109356.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 108250.00 | 107877.37 | 108370.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 108250.00 | 107877.37 | 108370.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 108250.00 | 107877.37 | 108370.29 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 108990.00 | 108465.86 | 108420.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 15:15:00 | 109188.00 | 108721.88 | 108558.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 10:15:00 | 108312.00 | 108660.40 | 108560.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 10:15:00 | 108312.00 | 108660.40 | 108560.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 108312.00 | 108660.40 | 108560.05 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 107995.00 | 108421.66 | 108462.40 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 09:15:00 | 108844.00 | 108459.91 | 108456.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 10:15:00 | 109774.00 | 108910.34 | 108702.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 12:15:00 | 108900.00 | 110252.96 | 109730.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 12:15:00 | 108900.00 | 110252.96 | 109730.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 108900.00 | 110252.96 | 109730.16 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 15:15:00 | 107830.00 | 109131.96 | 109294.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 13:15:00 | 107426.00 | 108238.56 | 108749.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 108036.00 | 107552.91 | 107950.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 108036.00 | 107552.91 | 107950.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 108036.00 | 107552.91 | 107950.59 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 14:15:00 | 108500.00 | 108132.98 | 108124.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 108850.00 | 108327.27 | 108217.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 12:15:00 | 108132.00 | 108340.58 | 108255.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 12:15:00 | 108132.00 | 108340.58 | 108255.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 108132.00 | 108340.58 | 108255.89 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 107490.00 | 108071.07 | 108147.46 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 10:15:00 | 108513.00 | 107953.94 | 107952.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 12:15:00 | 109060.00 | 108310.20 | 108122.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 110908.00 | 111133.24 | 110404.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 111565.00 | 111385.13 | 111093.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 111565.00 | 111385.13 | 111093.29 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 10:15:00 | 111474.00 | 111808.46 | 111852.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 09:15:00 | 110976.00 | 111437.01 | 111625.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 11:15:00 | 111518.00 | 111409.21 | 111577.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 11:15:00 | 111518.00 | 111409.21 | 111577.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 11:15:00 | 111518.00 | 111409.21 | 111577.16 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 10:15:00 | 111901.00 | 111616.85 | 111604.85 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 11:15:00 | 111368.00 | 111567.08 | 111583.32 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 12:15:00 | 112247.00 | 111703.06 | 111643.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 13:15:00 | 113062.00 | 111974.85 | 111772.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 15:15:00 | 119008.00 | 119271.22 | 118516.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 15:15:00 | 120051.00 | 120145.46 | 119871.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 120051.00 | 120145.46 | 119871.88 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 14:15:00 | 119480.00 | 119790.84 | 119792.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 15:15:00 | 119313.00 | 119695.27 | 119749.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 119399.00 | 119197.60 | 119398.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 119399.00 | 119197.60 | 119398.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 119399.00 | 119197.60 | 119398.02 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 118800.00 | 118338.14 | 118337.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 13:15:00 | 120208.00 | 118712.11 | 118507.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 13:15:00 | 120444.00 | 120466.63 | 119911.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 128103.00 | 129007.59 | 127530.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 128103.00 | 129007.59 | 127530.32 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 10:15:00 | 130900.00 | 131777.72 | 131801.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 11:15:00 | 130586.00 | 131539.38 | 131691.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 131832.00 | 131350.95 | 131544.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 14:15:00 | 131832.00 | 131350.95 | 131544.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 131832.00 | 131350.95 | 131544.01 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 132900.00 | 131764.60 | 131704.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 11:15:00 | 133210.00 | 132282.39 | 131964.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 136390.00 | 136442.82 | 135603.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 136390.00 | 136442.82 | 135603.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 136390.00 | 136442.82 | 135603.73 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 11:15:00 | 137707.00 | 140653.43 | 140835.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 10:15:00 | 136500.00 | 138961.96 | 139830.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 139598.00 | 137913.11 | 138746.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 139598.00 | 137913.11 | 138746.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 139598.00 | 137913.11 | 138746.78 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 12:15:00 | 143979.00 | 140108.50 | 139619.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 09:15:00 | 144138.00 | 142463.51 | 141768.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 142300.00 | 142684.33 | 142005.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 11:15:00 | 142300.00 | 142684.33 | 142005.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 142300.00 | 142684.33 | 142005.61 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 13:15:00 | 141088.00 | 141883.09 | 141969.75 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 10:15:00 | 143503.00 | 142157.60 | 142043.26 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 15:15:00 | 141753.00 | 141958.20 | 141982.58 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 143100.00 | 142186.56 | 142084.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 144188.00 | 143123.52 | 142657.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 13:15:00 | 143409.00 | 143417.10 | 142972.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 15:15:00 | 143181.00 | 143341.38 | 143013.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 15:15:00 | 143181.00 | 143341.38 | 143013.10 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 14:15:00 | 142296.00 | 142772.95 | 142834.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 141840.00 | 142622.69 | 142758.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 141750.00 | 139650.47 | 140849.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 141750.00 | 139650.47 | 140849.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 141750.00 | 139650.47 | 140849.12 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 13:15:00 | 142658.00 | 141565.02 | 141501.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 14:15:00 | 143914.00 | 142034.81 | 141720.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 12:15:00 | 142901.00 | 143013.78 | 142428.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 143788.00 | 143721.13 | 143003.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 143788.00 | 143721.13 | 143003.14 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 147800.00 | 149657.26 | 149752.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 146464.95 | 148090.95 | 148733.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 13:15:00 | 146393.50 | 146154.12 | 146822.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 146156.95 | 146154.69 | 146762.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 146156.95 | 146154.69 | 146762.03 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 15:15:00 | 146230.00 | 145859.49 | 145819.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 09:15:00 | 146616.80 | 146010.95 | 145891.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 145799.20 | 145968.60 | 145883.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 10:15:00 | 145799.20 | 145968.60 | 145883.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 145799.20 | 145968.60 | 145883.20 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 144999.00 | 145715.49 | 145800.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 10:15:00 | 144150.00 | 144934.89 | 145266.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 10:15:00 | 144795.91 | 144426.98 | 144771.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 10:15:00 | 144795.91 | 144426.98 | 144771.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 144795.91 | 144426.98 | 144771.61 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 133150.05 | 132020.03 | 131975.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 133553.41 | 132621.95 | 132330.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 12:15:00 | 136699.41 | 136707.05 | 135790.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 09:15:00 | 136600.00 | 136484.52 | 135958.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 136600.00 | 136484.52 | 135958.58 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 09:15:00 | 135599.95 | 135786.72 | 135806.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 134279.00 | 135360.22 | 135575.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 133700.66 | 133683.55 | 134444.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 133594.95 | 133366.66 | 133883.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 133594.95 | 133366.66 | 133883.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 14:00:00 | 132300.00 | 132817.13 | 133168.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 14:15:00 | 129718.80 | 129188.00 | 129183.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 14:15:00 | 129718.80 | 129188.00 | 129183.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 129948.75 | 129422.07 | 129295.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 09:15:00 | 132742.30 | 132887.01 | 131961.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 10:00:00 | 132742.30 | 132887.01 | 131961.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 130472.90 | 133069.14 | 132641.82 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 129700.00 | 131919.20 | 132163.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 12:15:00 | 128660.40 | 131267.44 | 131845.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 125979.95 | 125118.86 | 126550.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 125979.95 | 125118.86 | 126550.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 126623.15 | 125560.55 | 126303.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 15:15:00 | 125802.00 | 126424.52 | 126484.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 10:15:00 | 126951.00 | 126512.16 | 126502.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 126951.00 | 126512.16 | 126502.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 14:15:00 | 127546.20 | 126857.89 | 126682.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 126146.90 | 126850.11 | 126717.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 126146.90 | 126850.11 | 126717.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 126146.90 | 126850.11 | 126717.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:00:00 | 126146.90 | 126850.11 | 126717.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 126720.05 | 126824.10 | 126718.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 12:15:00 | 127129.95 | 126798.72 | 126716.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 129337.95 | 130399.20 | 130518.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 129337.95 | 130399.20 | 130518.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 129139.85 | 130147.33 | 130393.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 126568.20 | 126529.72 | 127584.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 10:45:00 | 126600.00 | 126529.72 | 127584.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 128499.55 | 126648.38 | 127152.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 124885.40 | 126943.06 | 127115.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 118641.13 | 123982.39 | 125510.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 125360.80 | 123721.62 | 124840.55 | SL hit (close>ema200) qty=0.50 sl=123721.62 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 127283.70 | 125547.12 | 125445.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 128093.65 | 126056.43 | 125685.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 12:15:00 | 126806.45 | 126909.82 | 126326.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 13:00:00 | 126806.45 | 126909.82 | 126326.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 126250.05 | 126763.18 | 126359.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 126250.05 | 126763.18 | 126359.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 126300.00 | 126670.55 | 126354.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 126676.50 | 126670.55 | 126354.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 126979.60 | 126737.27 | 126464.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:30:00 | 126268.00 | 126737.27 | 126464.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 126600.00 | 126958.45 | 126720.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 126600.00 | 126958.45 | 126720.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 126599.25 | 126886.61 | 126709.77 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 15:15:00 | 126348.00 | 126675.66 | 126716.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 11:15:00 | 126040.00 | 126467.34 | 126604.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 09:15:00 | 126379.90 | 126295.42 | 126447.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 09:15:00 | 126379.90 | 126295.42 | 126447.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 126379.90 | 126295.42 | 126447.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 10:45:00 | 126021.05 | 126218.33 | 126398.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 12:30:00 | 126066.60 | 126193.49 | 126355.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 13:15:00 | 126072.45 | 126193.49 | 126355.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 14:45:00 | 126054.05 | 126152.66 | 126307.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 125863.60 | 126063.38 | 126238.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 14:15:00 | 125605.00 | 125946.28 | 126124.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 125599.95 | 125877.01 | 126077.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:00:00 | 125376.05 | 125762.92 | 125902.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 13:15:00 | 126320.85 | 125804.76 | 125744.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 126320.85 | 125804.76 | 125744.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 126950.00 | 126252.04 | 126006.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 126031.80 | 126207.99 | 126008.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 12:15:00 | 126031.80 | 126207.99 | 126008.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 126031.80 | 126207.99 | 126008.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 126031.80 | 126207.99 | 126008.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 126555.05 | 126277.40 | 126058.39 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 125065.50 | 125858.49 | 125895.04 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 126221.95 | 125940.50 | 125913.80 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 125677.05 | 125887.81 | 125892.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 125030.25 | 125716.30 | 125813.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 126134.60 | 125797.75 | 125833.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 126134.60 | 125797.75 | 125833.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 126134.60 | 125797.75 | 125833.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:45:00 | 126309.40 | 125797.75 | 125833.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 125935.40 | 125825.28 | 125843.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 125700.00 | 125820.22 | 125839.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:15:00 | 125638.95 | 125820.22 | 125839.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 126224.80 | 125872.30 | 125857.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 126224.80 | 125872.30 | 125857.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 126974.40 | 126092.72 | 125959.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 126050.00 | 126414.57 | 126200.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 126050.00 | 126414.57 | 126200.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 126050.00 | 126414.57 | 126200.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 126050.00 | 126414.57 | 126200.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 126300.00 | 126391.66 | 126209.56 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 125350.00 | 125983.55 | 126056.18 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 127498.90 | 126187.91 | 126059.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 130399.70 | 127382.55 | 126666.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 10:15:00 | 128080.85 | 128191.45 | 127349.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 10:30:00 | 128064.50 | 128191.45 | 127349.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 128616.80 | 129079.77 | 128475.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 128616.80 | 129079.77 | 128475.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 129000.00 | 129063.82 | 128523.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 129312.05 | 129063.82 | 128523.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 128463.00 | 128832.98 | 128542.79 | SL hit (close<static) qty=1.00 sl=128523.55 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 130087.00 | 130266.00 | 130270.29 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 132946.80 | 130685.69 | 130449.54 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 128100.00 | 131409.64 | 131616.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 127983.85 | 129270.87 | 130341.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 129157.35 | 128996.67 | 130015.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 129157.35 | 128996.67 | 130015.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 130640.00 | 129325.34 | 130072.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 130640.00 | 129325.34 | 130072.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 131000.00 | 129660.27 | 130156.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 131000.00 | 129660.27 | 130156.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 131660.00 | 130593.49 | 130482.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 134550.00 | 132377.49 | 131543.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 137125.30 | 137183.30 | 135570.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:15:00 | 136786.09 | 137183.30 | 135570.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 140619.66 | 141142.21 | 140524.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 140639.84 | 141142.21 | 140524.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 140350.00 | 140983.76 | 140508.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 140500.00 | 140983.76 | 140508.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 140386.41 | 140864.29 | 140497.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:45:00 | 139969.84 | 140864.29 | 140497.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 140298.00 | 140751.03 | 140479.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 140500.00 | 140751.03 | 140479.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 140326.25 | 140536.30 | 140420.84 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 139468.95 | 140322.83 | 140334.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 138348.05 | 139635.98 | 139991.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 138665.55 | 137270.74 | 138166.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 138665.55 | 137270.74 | 138166.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 138665.55 | 137270.74 | 138166.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 138665.55 | 137270.74 | 138166.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 137870.95 | 137390.78 | 138139.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 137080.45 | 137247.58 | 137900.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 13:15:00 | 141273.66 | 135156.49 | 135673.69 | SL hit (close>static) qty=1.00 sl=138672.84 alert=retest2 |

### Cycle 87 — BUY (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 14:15:00 | 140731.05 | 136271.40 | 136133.45 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 136138.84 | 136530.70 | 136540.90 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 13:15:00 | 137261.75 | 136673.52 | 136603.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 138111.20 | 137099.85 | 136828.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 137488.66 | 137515.80 | 137147.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 13:15:00 | 137488.66 | 137515.80 | 137147.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 137488.66 | 137515.80 | 137147.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 137488.66 | 137515.80 | 137147.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 137528.41 | 137512.76 | 137209.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 137489.59 | 137512.76 | 137209.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 137094.00 | 137429.01 | 137199.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 137200.00 | 137429.01 | 137199.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 137105.05 | 137364.22 | 137190.66 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 135976.55 | 136926.52 | 137029.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 135770.00 | 136695.22 | 136915.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 136225.30 | 136009.51 | 136386.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 15:00:00 | 136225.30 | 136009.51 | 136386.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 135623.84 | 135899.17 | 136268.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 135450.05 | 135899.17 | 136268.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:00:00 | 135380.00 | 135699.47 | 136107.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 10:30:00 | 135486.05 | 135306.23 | 135686.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 11:30:00 | 135417.59 | 135324.98 | 135660.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 135917.30 | 135443.45 | 135684.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 135896.91 | 135443.45 | 135684.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 135710.25 | 135496.81 | 135686.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:15:00 | 135870.75 | 135496.81 | 135686.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 135997.00 | 135596.85 | 135714.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:30:00 | 135900.00 | 135596.85 | 135714.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 135555.00 | 135588.48 | 135700.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 136646.95 | 135588.48 | 135700.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 137350.00 | 135940.78 | 135850.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 137350.00 | 135940.78 | 135850.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 12:15:00 | 137579.00 | 136619.94 | 136213.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 12:15:00 | 138655.05 | 138910.52 | 138198.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:45:00 | 139445.20 | 138919.92 | 138416.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 138550.00 | 139089.45 | 138643.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-26 12:15:00 | 138550.00 | 139089.45 | 138643.67 | SL hit (close<ema400) qty=1.00 sl=138643.67 alert=retest1 |

### Cycle 92 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 137999.59 | 138740.42 | 138779.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 136912.20 | 138044.10 | 138408.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 135219.84 | 135179.67 | 136337.97 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 12:00:00 | 134549.00 | 135015.57 | 136060.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 134853.50 | 134870.67 | 135572.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 134630.09 | 134870.67 | 135572.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:00:00 | 134706.20 | 134837.78 | 135493.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 134596.25 | 134750.71 | 135394.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 134999.41 | 134569.40 | 134792.69 | SL hit (close>ema400) qty=1.00 sl=134792.69 alert=retest1 |

### Cycle 93 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 135390.00 | 134932.09 | 134896.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 136086.50 | 135237.83 | 135046.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 135505.09 | 135833.71 | 135531.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 135505.09 | 135833.71 | 135531.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 135505.09 | 135833.71 | 135531.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 135505.09 | 135833.71 | 135531.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 135279.95 | 135722.96 | 135508.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 135318.16 | 135722.96 | 135508.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 135307.00 | 135639.77 | 135490.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 135629.95 | 135639.77 | 135490.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 135000.00 | 135511.81 | 135445.64 | SL hit (close<static) qty=1.00 sl=135147.91 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 134228.00 | 135185.19 | 135304.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 133589.41 | 134708.40 | 135057.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 135121.95 | 134477.75 | 134725.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 135121.95 | 134477.75 | 134725.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 135121.95 | 134477.75 | 134725.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 135000.00 | 134477.75 | 134725.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 135489.00 | 134680.00 | 134795.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 135489.00 | 134680.00 | 134795.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 136001.00 | 135009.34 | 134929.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 136549.80 | 135697.27 | 135334.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 135800.00 | 135922.77 | 135546.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 135800.00 | 135922.77 | 135546.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 135282.05 | 135794.63 | 135522.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 135282.05 | 135794.63 | 135522.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 135775.30 | 135790.76 | 135545.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 136000.30 | 135790.76 | 135545.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:15:00 | 135923.16 | 135762.62 | 135554.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 135999.55 | 135936.19 | 135718.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 136300.00 | 135870.32 | 135726.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 136649.00 | 136026.06 | 135809.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:00:00 | 137189.16 | 136258.68 | 135935.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 135550.91 | 136374.47 | 136411.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 135550.91 | 136374.47 | 136411.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 135102.41 | 135576.59 | 135931.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 135527.20 | 134757.29 | 135224.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 135527.20 | 134757.29 | 135224.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 135527.20 | 134757.29 | 135224.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 135810.16 | 134757.29 | 135224.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 136000.00 | 135005.84 | 135294.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 136000.00 | 135005.84 | 135294.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 134890.09 | 134982.69 | 135258.14 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 136300.00 | 135448.83 | 135420.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 136500.00 | 135659.06 | 135518.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 11:15:00 | 135866.55 | 135877.32 | 135665.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 135866.55 | 135877.32 | 135665.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 135866.55 | 135877.32 | 135665.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 135795.66 | 135877.32 | 135665.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 135620.95 | 135917.47 | 135726.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 135292.41 | 135917.47 | 135726.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 136006.95 | 135935.37 | 135751.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:45:00 | 135233.95 | 135935.37 | 135751.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 138038.34 | 138633.67 | 138105.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:45:00 | 137900.00 | 138633.67 | 138105.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 138400.00 | 138586.94 | 138132.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:30:00 | 138119.84 | 138586.94 | 138132.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 141747.09 | 139285.37 | 138599.77 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 138600.00 | 139540.11 | 139588.29 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 139963.00 | 139649.09 | 139627.89 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 138568.95 | 139651.16 | 139668.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 137368.95 | 138758.28 | 139218.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 132156.50 | 131981.36 | 133135.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 132156.50 | 131981.36 | 133135.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 133051.84 | 132230.44 | 133050.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:45:00 | 132350.05 | 132706.78 | 132905.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 130926.55 | 130289.31 | 130240.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 130926.55 | 130289.31 | 130240.23 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 129508.30 | 130133.11 | 130173.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 128800.00 | 129866.49 | 130048.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 123891.10 | 123316.43 | 124327.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:00:00 | 123891.10 | 123316.43 | 124327.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 123799.85 | 123413.11 | 124279.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:30:00 | 123435.00 | 123430.08 | 124141.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 18:15:00 | 123379.95 | 122668.16 | 122636.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 123379.95 | 122668.16 | 122636.79 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 122392.00 | 122612.93 | 122614.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 11:15:00 | 121636.05 | 122414.57 | 122523.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 15:15:00 | 119845.00 | 119769.00 | 120684.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:15:00 | 120499.00 | 119769.00 | 120684.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 120368.35 | 119888.87 | 120656.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:45:00 | 120301.10 | 119888.87 | 120656.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 120348.35 | 119980.77 | 120628.04 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 15:15:00 | 121840.00 | 121028.99 | 120956.41 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 120521.80 | 120909.41 | 120924.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 119290.55 | 120585.64 | 120775.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 123936.00 | 120403.29 | 120442.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 123936.00 | 120403.29 | 120442.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 123936.00 | 120403.29 | 120442.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:45:00 | 124215.15 | 120403.29 | 120442.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 122865.05 | 120895.64 | 120663.06 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 120878.95 | 121454.14 | 121507.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 120499.00 | 121097.07 | 121310.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 121400.00 | 121097.40 | 121270.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 121400.00 | 121097.40 | 121270.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 121400.00 | 121097.40 | 121270.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 121400.00 | 121097.40 | 121270.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 120500.05 | 120977.93 | 121200.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 120925.75 | 120977.93 | 121200.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 120601.00 | 120744.65 | 120989.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 120312.05 | 120744.65 | 120989.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:00:00 | 119977.50 | 120591.22 | 120897.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:00:00 | 120325.35 | 120454.78 | 120751.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 121369.70 | 120888.38 | 120857.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 121369.70 | 120888.38 | 120857.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 121981.50 | 121107.00 | 120959.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 122012.30 | 122280.76 | 121721.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:30:00 | 122404.15 | 122280.76 | 121721.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 123219.20 | 123124.56 | 122628.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 12:15:00 | 123437.90 | 123124.56 | 122628.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:45:00 | 123690.90 | 124331.78 | 124251.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 123590.00 | 124175.66 | 124237.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 123590.00 | 124175.66 | 124237.62 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 125156.85 | 124224.69 | 124190.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 125334.05 | 124880.36 | 124646.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 130691.90 | 130760.45 | 129393.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 15:00:00 | 130691.90 | 130760.45 | 129393.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 132417.25 | 132416.35 | 132063.06 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 131445.80 | 131941.54 | 131955.89 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 132590.59 | 132058.97 | 132006.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 133300.00 | 132498.44 | 132235.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 11:15:00 | 132666.20 | 132682.03 | 132424.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 132558.16 | 132639.08 | 132467.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 132558.16 | 132639.08 | 132467.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 132700.00 | 132639.08 | 132467.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 131822.00 | 132485.41 | 132428.06 | SL hit (close<static) qty=1.00 sl=132300.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 131605.50 | 132309.43 | 132353.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 130952.00 | 131711.07 | 132026.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 129908.95 | 129874.33 | 130519.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 129908.95 | 129874.33 | 130519.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 129685.00 | 129808.73 | 130375.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 130549.10 | 129808.73 | 130375.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 130205.60 | 129114.02 | 129591.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 130205.60 | 129114.02 | 129591.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 130195.25 | 129330.27 | 129646.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 130498.50 | 129330.27 | 129646.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 130014.60 | 129530.23 | 129686.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:45:00 | 130092.15 | 129530.23 | 129686.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 15:15:00 | 130226.00 | 129808.84 | 129794.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 130691.85 | 130082.68 | 129927.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 12:15:00 | 130054.60 | 130202.47 | 130016.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 130054.60 | 130202.47 | 130016.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 130054.60 | 130202.47 | 130016.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 130054.60 | 130202.47 | 130016.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 130140.15 | 130190.00 | 130027.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 130419.75 | 130164.50 | 130045.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 10:00:00 | 130457.30 | 130223.06 | 130082.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 11:00:00 | 130900.00 | 130358.45 | 130157.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:45:00 | 130425.85 | 131152.54 | 130945.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 130842.45 | 131090.52 | 130936.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 129655.00 | 130619.44 | 130745.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 129655.00 | 130619.44 | 130745.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 14:15:00 | 129485.05 | 130099.56 | 130349.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 14:15:00 | 129319.90 | 129036.40 | 129552.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-02 14:45:00 | 129315.40 | 129036.40 | 129552.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 128905.35 | 129034.61 | 129463.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 10:30:00 | 128526.50 | 128793.35 | 129315.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 09:15:00 | 122100.17 | 123871.43 | 125681.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 120383.85 | 120316.40 | 121740.28 | SL hit (close>ema200) qty=0.50 sl=120316.40 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 112487.05 | 111632.65 | 111585.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 13:15:00 | 112750.05 | 111973.13 | 111755.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 10:15:00 | 111877.60 | 112233.20 | 111981.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 10:15:00 | 111877.60 | 112233.20 | 111981.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 111877.60 | 112233.20 | 111981.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 111877.60 | 112233.20 | 111981.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 112416.25 | 112269.81 | 112020.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 09:30:00 | 112611.90 | 112385.10 | 112160.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 113000.10 | 112641.12 | 112428.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 13:00:00 | 113695.10 | 114667.77 | 114519.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 13:15:00 | 113309.05 | 114396.03 | 114409.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 113309.05 | 114396.03 | 114409.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 112946.65 | 113924.75 | 114155.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 109150.70 | 109125.74 | 110434.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 109150.70 | 109125.74 | 110434.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 110658.45 | 109432.28 | 110455.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 110658.45 | 109432.28 | 110455.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 110445.25 | 109634.87 | 110454.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 110464.90 | 109634.87 | 110454.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 109409.85 | 109589.87 | 110359.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:15:00 | 109196.00 | 109589.87 | 110359.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 109038.00 | 109590.51 | 109893.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 109125.55 | 109497.52 | 109823.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:30:00 | 109150.00 | 109448.12 | 109771.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 109550.00 | 109381.60 | 109679.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:30:00 | 109881.50 | 109381.60 | 109679.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 108695.45 | 109228.11 | 109557.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 109950.00 | 109428.84 | 109374.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 109950.00 | 109428.84 | 109374.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 110000.00 | 109543.07 | 109431.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 108982.25 | 109643.39 | 109556.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 108982.25 | 109643.39 | 109556.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 108982.25 | 109643.39 | 109556.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 108982.25 | 109643.39 | 109556.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 109314.00 | 109577.51 | 109534.42 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 109161.15 | 109494.24 | 109500.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 108299.00 | 109197.89 | 109355.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 10:15:00 | 109219.30 | 109202.17 | 109343.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 10:15:00 | 109219.30 | 109202.17 | 109343.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 109219.30 | 109202.17 | 109343.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 109108.50 | 109202.17 | 109343.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 108700.00 | 108922.44 | 109136.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 109092.15 | 108960.23 | 109134.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 108392.70 | 108846.72 | 109066.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 108288.85 | 108747.80 | 109001.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:00:00 | 108199.50 | 108627.64 | 108902.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 12:15:00 | 102874.41 | 104126.98 | 104912.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 13:15:00 | 102789.52 | 104051.58 | 104807.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 103817.50 | 103615.48 | 104384.31 | SL hit (close>ema200) qty=0.50 sl=103615.48 alert=retest2 |

### Cycle 121 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 105800.00 | 104824.30 | 104718.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 107184.15 | 105296.27 | 104942.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 107468.10 | 107620.10 | 106756.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 107468.10 | 107620.10 | 106756.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 106680.00 | 107276.30 | 106946.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 106680.00 | 107276.30 | 106946.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 106863.90 | 107193.82 | 106939.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 106863.90 | 107193.82 | 106939.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 107220.40 | 107199.13 | 106964.84 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 106501.00 | 106796.29 | 106828.52 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 106853.70 | 106748.20 | 106739.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 107430.35 | 106887.98 | 106804.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 09:15:00 | 106589.65 | 106864.77 | 106811.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 106589.65 | 106864.77 | 106811.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 106589.65 | 106864.77 | 106811.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 106589.65 | 106864.77 | 106811.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 106152.30 | 106722.27 | 106751.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 105982.65 | 106574.35 | 106681.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 11:15:00 | 105924.95 | 105857.67 | 106178.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 11:30:00 | 105880.00 | 105857.67 | 106178.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 106700.05 | 105829.75 | 106015.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 106607.00 | 105829.75 | 106015.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 107020.30 | 106067.86 | 106106.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 107277.55 | 106067.86 | 106106.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 107270.05 | 106308.29 | 106212.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 107400.00 | 106526.64 | 106320.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 15:15:00 | 114115.00 | 114179.75 | 113345.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:15:00 | 115013.00 | 114179.75 | 113345.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:45:00 | 114656.60 | 114288.80 | 113471.01 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 114008.70 | 114556.93 | 114026.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 114008.70 | 114556.93 | 114026.31 | SL hit (close<ema400) qty=1.00 sl=114026.31 alert=retest1 |

### Cycle 126 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 113050.90 | 113713.69 | 113742.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 112350.00 | 113049.74 | 113347.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 113800.20 | 113088.86 | 113282.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 113800.20 | 113088.86 | 113282.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 113800.20 | 113088.86 | 113282.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 114005.00 | 113088.86 | 113282.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 113200.00 | 113111.09 | 113275.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 113133.60 | 113111.09 | 113275.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 113124.00 | 113136.70 | 113260.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 114199.80 | 113393.79 | 113327.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 114199.80 | 113393.79 | 113327.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 114653.40 | 113887.43 | 113590.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 114201.15 | 114280.07 | 113874.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 114201.15 | 114280.07 | 113874.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 114201.15 | 114280.07 | 113874.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 114971.65 | 114502.84 | 114141.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 112901.00 | 114242.45 | 114089.76 | SL hit (close<static) qty=1.00 sl=113800.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 112999.90 | 113859.15 | 113933.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 109505.35 | 112703.36 | 113335.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 112239.90 | 111123.89 | 111957.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 112239.90 | 111123.89 | 111957.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 112239.90 | 111123.89 | 111957.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 111450.00 | 111976.15 | 112112.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 113235.40 | 112276.22 | 112227.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 113235.40 | 112276.22 | 112227.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 116159.65 | 113340.72 | 112765.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 127515.00 | 127652.19 | 126303.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 13:45:00 | 127540.00 | 127652.19 | 126303.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 127865.00 | 128712.17 | 128083.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 127865.00 | 128712.17 | 128083.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 128575.00 | 128684.74 | 128128.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 128310.00 | 128684.74 | 128128.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 135055.00 | 135861.18 | 135136.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 135055.00 | 135861.18 | 135136.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 134990.00 | 135686.95 | 135123.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 134770.00 | 135686.95 | 135123.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 135055.00 | 135560.56 | 135117.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 135055.00 | 135560.56 | 135117.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 135030.00 | 135454.45 | 135109.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 135605.00 | 135212.45 | 135056.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 134485.00 | 135066.96 | 135004.14 | SL hit (close<static) qty=1.00 sl=134875.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 134365.00 | 134926.56 | 134946.03 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 140925.00 | 136069.20 | 135458.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 141885.00 | 138920.35 | 137154.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 138735.00 | 139521.35 | 138077.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 138735.00 | 139521.35 | 138077.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 136335.00 | 138614.73 | 138001.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 136335.00 | 138614.73 | 138001.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 136485.00 | 138188.79 | 137863.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 137630.00 | 138188.79 | 137863.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 140130.00 | 140673.96 | 140727.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 140130.00 | 140673.96 | 140727.00 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 142045.00 | 140712.62 | 140552.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 142125.00 | 140995.09 | 140695.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 141655.00 | 141704.21 | 141235.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:45:00 | 141800.00 | 141704.21 | 141235.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 144615.00 | 145336.31 | 144653.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 144615.00 | 145336.31 | 144653.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 144315.00 | 145132.05 | 144622.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 144315.00 | 145132.05 | 144622.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 143840.00 | 144873.64 | 144551.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 143840.00 | 144873.64 | 144551.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 142695.00 | 144080.70 | 144244.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 141990.00 | 142974.10 | 143469.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 139590.00 | 139308.22 | 140367.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:30:00 | 140085.00 | 139308.22 | 140367.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 140330.00 | 139512.58 | 140363.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 140330.00 | 139512.58 | 140363.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 140695.00 | 139749.06 | 140393.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 139835.00 | 139646.80 | 140239.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 140050.00 | 138433.86 | 138444.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 140080.00 | 138433.86 | 138444.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 139995.00 | 138746.09 | 138585.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 139995.00 | 138746.09 | 138585.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 140300.00 | 139056.87 | 138741.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 138995.00 | 139181.80 | 138861.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 138995.00 | 139181.80 | 138861.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 138995.00 | 139181.80 | 138861.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 141010.00 | 139382.55 | 139109.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 138530.00 | 139069.07 | 139101.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 138530.00 | 139069.07 | 139101.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 137445.00 | 138744.26 | 138951.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 138235.00 | 137876.78 | 138279.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 138235.00 | 137876.78 | 138279.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 138235.00 | 137876.78 | 138279.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 137835.00 | 137975.51 | 138232.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 137295.00 | 136686.78 | 136664.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 137295.00 | 136686.78 | 136664.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 137830.00 | 137120.83 | 136885.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 137620.00 | 137893.31 | 137520.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 138315.00 | 137893.31 | 137520.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 136415.00 | 137597.65 | 137419.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 136415.00 | 137597.65 | 137419.88 | SL hit (close<ema400) qty=1.00 sl=137419.88 alert=retest1 |

### Cycle 138 — SELL (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 12:15:00 | 136865.00 | 137273.88 | 137301.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 135000.00 | 136719.28 | 137038.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 136025.00 | 135723.68 | 136140.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 136025.00 | 135723.68 | 136140.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 136025.00 | 135723.68 | 136140.60 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 136755.00 | 136322.64 | 136315.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 137610.00 | 136703.69 | 136497.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 142830.00 | 142932.77 | 141150.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 142830.00 | 142932.77 | 141150.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 142725.00 | 143457.70 | 142388.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 142725.00 | 143457.70 | 142388.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 142005.00 | 143167.16 | 142353.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 143230.00 | 143167.16 | 142353.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:00:00 | 142915.00 | 142698.35 | 142357.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 143215.00 | 142619.36 | 142411.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 143690.00 | 144325.02 | 144336.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 143690.00 | 144325.02 | 144336.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 142940.00 | 143929.13 | 144143.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 143240.00 | 143130.54 | 143598.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 15:00:00 | 143240.00 | 143130.54 | 143598.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 143965.00 | 143196.55 | 143467.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 143965.00 | 143196.55 | 143467.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 144275.00 | 143412.24 | 143540.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 144300.00 | 143412.24 | 143540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 144905.00 | 143876.83 | 143739.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 146500.00 | 144421.17 | 144012.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 147590.00 | 148461.07 | 146885.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 147590.00 | 148461.07 | 146885.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 148220.00 | 148412.86 | 147007.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 149860.00 | 147910.86 | 147511.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 149030.00 | 148164.37 | 147815.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 148995.00 | 148173.49 | 147851.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 148480.00 | 150271.01 | 150330.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 148480.00 | 150271.01 | 150330.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 148000.00 | 149234.87 | 149768.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 14:15:00 | 148585.00 | 148258.74 | 148910.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 15:00:00 | 148585.00 | 148258.74 | 148910.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 149750.00 | 148595.60 | 148952.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 150070.00 | 148595.60 | 148952.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 149285.00 | 148733.48 | 148983.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 148855.00 | 148773.78 | 148978.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 149095.00 | 148827.03 | 148984.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:00:00 | 148950.00 | 148736.90 | 148858.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 148990.00 | 148859.62 | 148896.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 149035.00 | 148894.69 | 148908.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 149170.00 | 148894.69 | 148908.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 149945.00 | 149104.76 | 149002.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 151710.00 | 149749.04 | 149321.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 150165.00 | 150493.05 | 150023.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 150165.00 | 150493.05 | 150023.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 150165.00 | 150493.05 | 150023.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 150130.00 | 150493.05 | 150023.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 150245.00 | 150443.44 | 150043.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 150010.00 | 150443.44 | 150043.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 150105.00 | 150375.75 | 150048.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 150165.00 | 150375.75 | 150048.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 149795.00 | 150259.60 | 150025.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 149870.00 | 150259.60 | 150025.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 150040.00 | 150215.68 | 150027.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 149835.00 | 150215.68 | 150027.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 149875.00 | 150147.55 | 150013.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 149875.00 | 150147.55 | 150013.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 148755.00 | 149869.04 | 149898.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 148310.00 | 149291.31 | 149599.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 149410.00 | 148865.71 | 149290.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 149410.00 | 148865.71 | 149290.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 149410.00 | 148865.71 | 149290.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 149410.00 | 148865.71 | 149290.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 149240.00 | 148940.57 | 149285.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 149415.00 | 148940.57 | 149285.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 149485.00 | 149049.45 | 149303.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 149485.00 | 149049.45 | 149303.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 148700.00 | 148979.56 | 149248.79 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 149530.00 | 149342.72 | 149328.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 149825.00 | 149487.54 | 149400.11 | Break + close above crossover candle high |

### Cycle 146 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 148000.00 | 149352.66 | 149376.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 147695.00 | 149021.13 | 149223.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 11:15:00 | 145360.00 | 145056.13 | 145856.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:00:00 | 145360.00 | 145056.13 | 145856.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 146070.00 | 145286.74 | 145766.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 146070.00 | 145286.74 | 145766.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 145600.00 | 145349.39 | 145751.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 146860.00 | 145349.39 | 145751.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 146000.00 | 145479.51 | 145774.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:00:00 | 145725.00 | 145528.61 | 145769.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 146895.00 | 145990.11 | 145949.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 146895.00 | 145990.11 | 145949.31 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 144825.00 | 145919.29 | 145949.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 144605.00 | 145466.94 | 145726.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 140860.00 | 139130.88 | 139826.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 140860.00 | 139130.88 | 139826.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 140860.00 | 139130.88 | 139826.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 140860.00 | 139130.88 | 139826.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 143715.00 | 140047.70 | 140180.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 144175.00 | 140047.70 | 140180.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 143400.00 | 140718.16 | 140472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 144945.00 | 141563.53 | 140879.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 148000.00 | 148009.21 | 146868.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:00:00 | 148000.00 | 148009.21 | 146868.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 147410.00 | 147896.65 | 147180.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 147410.00 | 147896.65 | 147180.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 147210.00 | 147759.32 | 147183.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 147210.00 | 147759.32 | 147183.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 147220.00 | 147651.46 | 147186.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:30:00 | 147125.00 | 147651.46 | 147186.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 147190.00 | 147559.17 | 147187.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 147375.00 | 147643.33 | 147259.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 146605.00 | 147370.07 | 147225.76 | SL hit (close<static) qty=1.00 sl=147045.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 146670.00 | 147066.70 | 147116.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 145325.00 | 146706.09 | 146943.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 141920.00 | 141369.94 | 142412.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 141920.00 | 141369.94 | 142412.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 141920.00 | 141369.94 | 142412.37 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 145000.00 | 142845.84 | 142769.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 148505.00 | 144312.74 | 143470.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 150585.00 | 150707.94 | 148247.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:15:00 | 150325.00 | 150707.94 | 148247.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 149700.00 | 150230.47 | 149581.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 149140.00 | 150230.47 | 149581.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 149100.00 | 150004.38 | 149537.49 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 148295.00 | 149253.34 | 149310.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 147100.00 | 148640.54 | 149011.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 11:15:00 | 147770.00 | 147714.66 | 148199.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:30:00 | 147820.00 | 147714.66 | 148199.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 148600.00 | 147945.38 | 148223.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 148595.00 | 147945.38 | 148223.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 149395.00 | 148235.31 | 148330.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 149395.00 | 148235.31 | 148330.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 149150.00 | 148418.25 | 148404.62 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 10:15:00 | 147850.00 | 148332.08 | 148369.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 11:15:00 | 147035.00 | 148072.66 | 148248.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 146930.00 | 146568.99 | 147260.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 146930.00 | 146568.99 | 147260.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 147650.00 | 146700.56 | 147007.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 147650.00 | 146700.56 | 147007.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 148040.00 | 146968.45 | 147101.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 148200.00 | 146968.45 | 147101.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 146760.00 | 146961.41 | 147077.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 146675.00 | 146919.13 | 147047.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:00:00 | 146280.00 | 146791.30 | 146977.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 148350.00 | 147104.43 | 147087.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 148350.00 | 147104.43 | 147087.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 151400.00 | 148321.91 | 147750.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 150405.00 | 150495.50 | 149458.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:30:00 | 150390.00 | 150495.50 | 149458.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 149850.00 | 150236.34 | 149832.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 149850.00 | 150236.34 | 149832.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 149490.00 | 150087.07 | 149801.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 149490.00 | 150087.07 | 149801.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 149450.00 | 149959.66 | 149769.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 149845.00 | 149655.78 | 149654.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 148770.00 | 149478.63 | 149574.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 148770.00 | 149478.63 | 149574.34 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 150100.00 | 149623.24 | 149601.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 151345.00 | 150213.21 | 149895.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 154630.00 | 154655.59 | 153229.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 153385.00 | 154201.14 | 153363.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 153385.00 | 154201.14 | 153363.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 153420.00 | 154201.14 | 153363.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 153440.00 | 154048.91 | 153370.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 153440.00 | 154048.91 | 153370.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 153800.00 | 153999.13 | 153409.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:15:00 | 153300.00 | 153999.13 | 153409.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 153300.00 | 153859.31 | 153399.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 152340.00 | 153859.31 | 153399.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 151810.00 | 153449.44 | 153254.89 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 151640.00 | 153087.56 | 153108.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 151025.00 | 152675.04 | 152918.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 147230.00 | 146751.72 | 147721.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 147230.00 | 146751.72 | 147721.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 147230.00 | 146751.72 | 147721.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 146330.00 | 146707.37 | 147613.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:15:00 | 146365.00 | 146707.37 | 147613.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 146430.00 | 146657.90 | 147508.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 148310.00 | 146862.84 | 147377.86 | SL hit (close>static) qty=1.00 sl=147855.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 149585.00 | 147677.22 | 147675.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 150555.00 | 148252.78 | 147937.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 151495.00 | 151845.72 | 150993.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 151495.00 | 151845.72 | 150993.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 154890.00 | 155666.34 | 155052.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:45:00 | 154460.00 | 155666.34 | 155052.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 154445.00 | 155422.07 | 154996.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 154445.00 | 155422.07 | 154996.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 155200.00 | 155355.72 | 155039.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 155200.00 | 155355.72 | 155039.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 155850.00 | 155479.26 | 155151.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 156830.00 | 155479.26 | 155151.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 156395.00 | 155652.41 | 155260.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 156390.00 | 155789.93 | 155358.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 156400.00 | 155888.94 | 155442.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 156450.00 | 157187.32 | 156681.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 156450.00 | 157187.32 | 156681.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 155600.00 | 156869.86 | 156583.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 155600.00 | 156869.86 | 156583.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 156300.00 | 156755.89 | 156557.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 155185.00 | 156233.13 | 156350.74 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 157900.00 | 156155.23 | 156034.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 159145.00 | 157270.88 | 156618.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 161000.00 | 161640.98 | 160286.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 162850.00 | 161872.78 | 160515.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 160590.00 | 161541.38 | 160595.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 160590.00 | 161541.38 | 160595.57 | SL hit (close<ema400) qty=1.00 sl=160595.57 alert=retest1 |

### Cycle 162 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 159315.00 | 160235.60 | 160326.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 158455.00 | 159670.86 | 160014.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 159145.00 | 158963.62 | 159482.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 159145.00 | 158963.62 | 159482.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 159145.00 | 158963.62 | 159482.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:45:00 | 158600.00 | 158986.73 | 159365.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:45:00 | 158500.00 | 159188.39 | 159355.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 158450.00 | 159034.71 | 159270.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 158255.00 | 158597.87 | 158917.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 158330.00 | 158544.30 | 158863.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 157885.00 | 158544.30 | 158863.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 158030.00 | 158513.44 | 158820.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 158260.00 | 158189.97 | 158435.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 158210.00 | 158232.98 | 158433.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 158205.00 | 158227.38 | 158412.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 159060.00 | 158485.86 | 158490.68 | SL hit (close>static) qty=1.00 sl=158890.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 158605.00 | 158148.56 | 158096.60 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 157725.00 | 158176.02 | 158183.17 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 158910.00 | 158260.02 | 158173.19 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 157650.00 | 158121.91 | 158143.50 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 158800.00 | 158257.53 | 158203.18 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 157920.00 | 158263.80 | 158298.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 155830.00 | 156989.46 | 157459.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 153150.00 | 152821.68 | 153820.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 153150.00 | 152821.68 | 153820.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 153150.00 | 152821.68 | 153820.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 153870.00 | 152821.68 | 153820.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 152655.00 | 152602.73 | 153202.77 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 157395.00 | 153852.66 | 153583.50 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 153600.00 | 154076.05 | 154117.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 152650.00 | 153719.94 | 153932.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 152610.00 | 152304.01 | 152780.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 152610.00 | 152304.01 | 152780.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 152100.00 | 152263.21 | 152718.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 154220.00 | 152263.21 | 152718.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 154800.00 | 152770.57 | 152907.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 154800.00 | 152770.57 | 152907.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 155135.00 | 153243.45 | 153110.22 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 152610.00 | 153372.25 | 153382.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 152250.00 | 153017.84 | 153211.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 152815.00 | 152693.09 | 152989.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 152815.00 | 152693.09 | 152989.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 152815.00 | 152693.09 | 152989.57 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 153540.00 | 153120.04 | 153114.11 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 152810.00 | 153152.45 | 153170.98 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 153200.00 | 153151.40 | 153149.28 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 152500.00 | 153021.12 | 153090.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 151365.00 | 152040.10 | 152407.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 152235.00 | 151831.53 | 152195.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 152235.00 | 151831.53 | 152195.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 152235.00 | 151831.53 | 152195.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 152235.00 | 151831.53 | 152195.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 152700.00 | 152005.23 | 152241.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 152465.00 | 152005.23 | 152241.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 152660.00 | 152136.18 | 152279.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:15:00 | 152720.00 | 152136.18 | 152279.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 153700.00 | 152448.94 | 152408.51 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 151765.00 | 152425.16 | 152477.65 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 152790.00 | 152555.04 | 152529.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 153200.00 | 152712.83 | 152607.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 152725.00 | 152729.21 | 152634.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 152725.00 | 152729.21 | 152634.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 152725.00 | 152729.21 | 152634.10 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 152400.00 | 152582.92 | 152601.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 152000.00 | 152466.34 | 152547.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 13:15:00 | 152300.00 | 152013.39 | 152203.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 152300.00 | 152013.39 | 152203.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 152300.00 | 152013.39 | 152203.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 152300.00 | 152013.39 | 152203.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 152215.00 | 152053.71 | 152204.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 152010.00 | 152053.71 | 152204.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 152375.00 | 152117.97 | 152220.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 153850.00 | 152117.97 | 152220.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 153995.00 | 152493.37 | 152381.57 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 152900.00 | 153310.58 | 153352.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 152125.00 | 153073.47 | 153240.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 149185.00 | 149166.82 | 150257.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:00:00 | 149185.00 | 149166.82 | 150257.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 150000.00 | 149163.45 | 149829.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 150000.00 | 149163.45 | 149829.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 152500.00 | 149830.76 | 150072.42 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 152895.00 | 150443.61 | 150329.02 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 151160.00 | 151611.95 | 151617.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 151005.00 | 151490.56 | 151561.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 150695.00 | 150518.52 | 150918.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 150695.00 | 150518.52 | 150918.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 149950.00 | 150291.46 | 150738.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 149500.00 | 150272.65 | 150389.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:45:00 | 149370.00 | 150093.12 | 150297.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 142025.00 | 143135.63 | 144437.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 141901.50 | 143135.63 | 144437.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 143195.00 | 143041.56 | 144061.00 | SL hit (close>ema200) qty=0.50 sl=143041.56 alert=retest2 |

### Cycle 185 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 133800.00 | 132957.37 | 132911.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 135180.00 | 133401.89 | 133117.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 135300.00 | 135630.01 | 134584.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:15:00 | 139100.00 | 135630.01 | 134584.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 135930.00 | 137182.44 | 136253.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 135930.00 | 137182.44 | 136253.20 | SL hit (close<ema400) qty=1.00 sl=136253.20 alert=retest1 |

### Cycle 186 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 134750.00 | 135742.50 | 135837.94 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 137175.00 | 136029.00 | 135959.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 143855.00 | 137915.61 | 136863.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 12:15:00 | 151045.00 | 151076.58 | 149122.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:00:00 | 151045.00 | 151076.58 | 149122.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 149170.00 | 150722.33 | 149589.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 149170.00 | 150722.33 | 149589.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 150000.00 | 150577.86 | 149626.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 151030.00 | 150668.29 | 149754.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 147920.00 | 149616.58 | 149687.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 147920.00 | 149616.58 | 149687.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 147105.00 | 148679.57 | 149205.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 147940.00 | 147806.66 | 148398.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 147940.00 | 147806.66 | 148398.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 147860.00 | 147887.46 | 148336.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 147155.00 | 148066.06 | 148271.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 146580.00 | 145419.99 | 145370.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 146580.00 | 145419.99 | 145370.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 147570.00 | 146054.79 | 145680.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 145880.00 | 146203.07 | 145824.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 145880.00 | 146203.07 | 145824.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 145880.00 | 146203.07 | 145824.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 145880.00 | 146203.07 | 145824.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 144695.00 | 145901.45 | 145721.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 144695.00 | 145901.45 | 145721.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 144245.00 | 145570.16 | 145587.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 14:15:00 | 143550.00 | 144693.12 | 145140.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 138990.00 | 136576.90 | 138277.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 138990.00 | 136576.90 | 138277.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 138990.00 | 136576.90 | 138277.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 138990.00 | 136576.90 | 138277.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 138045.00 | 136870.52 | 138256.36 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 138815.00 | 138683.49 | 138679.07 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 134970.00 | 138003.79 | 138375.99 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 137795.00 | 136999.14 | 136968.84 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 135640.00 | 137003.72 | 137004.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 135435.00 | 136371.30 | 136684.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 134200.00 | 133070.20 | 134175.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 134200.00 | 133070.20 | 134175.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 134200.00 | 133070.20 | 134175.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 134200.00 | 133070.20 | 134175.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 133060.00 | 133068.16 | 134073.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 132490.00 | 133871.14 | 133905.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 125865.50 | 128151.61 | 130155.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 125930.00 | 125759.66 | 127647.38 | SL hit (close>ema200) qty=0.50 sl=125759.66 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 130375.00 | 128251.76 | 128057.74 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 125890.00 | 128645.09 | 128669.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 124890.00 | 127894.07 | 128326.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 127400.00 | 127351.68 | 127929.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:00:00 | 127400.00 | 127351.68 | 127929.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 128545.00 | 127590.35 | 127985.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:30:00 | 129725.00 | 127590.35 | 127985.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 128800.00 | 127832.28 | 128059.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 131510.00 | 127832.28 | 128059.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 131105.00 | 128486.82 | 128336.00 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 126320.00 | 128333.96 | 128590.63 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 133010.00 | 128240.09 | 127718.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 134980.00 | 133052.40 | 131152.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 135155.00 | 136002.02 | 134515.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 135155.00 | 136002.02 | 134515.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 135155.00 | 136002.02 | 134515.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 135300.00 | 136002.02 | 134515.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 135270.00 | 135844.62 | 134578.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 136420.00 | 135025.09 | 134619.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 137670.00 | 139106.31 | 139179.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 137670.00 | 139106.31 | 139179.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 14:15:00 | 137265.00 | 138191.97 | 138672.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 15:15:00 | 132990.00 | 132525.00 | 133513.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 131700.00 | 132275.00 | 133310.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 133100.00 | 131311.09 | 132147.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 133100.00 | 131311.09 | 132147.76 | SL hit (close>ema400) qty=1.00 sl=132147.76 alert=retest1 |

### Cycle 201 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 130300.00 | 129617.96 | 129609.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 131780.00 | 130050.37 | 129806.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 13:15:00 | 130555.00 | 130740.72 | 130272.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 14:00:00 | 130555.00 | 130740.72 | 130272.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 129835.00 | 130559.58 | 130232.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 131135.00 | 130477.66 | 130224.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 130800.00 | 130468.13 | 130243.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 130665.00 | 130503.44 | 130319.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 14:00:00 | 132300.00 | 2024-04-25 14:15:00 | 129718.80 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2024-05-09 15:15:00 | 125802.00 | 2024-05-10 10:15:00 | 126951.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-05-13 12:15:00 | 127129.95 | 2024-05-28 13:15:00 | 129337.95 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2024-06-04 09:15:00 | 124885.40 | 2024-06-04 12:15:00 | 118641.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 124885.40 | 2024-06-05 09:15:00 | 125360.80 | STOP_HIT | 0.50 | -0.38% |
| SELL | retest2 | 2024-06-05 12:15:00 | 126599.90 | 2024-06-05 13:15:00 | 127283.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-06-13 10:45:00 | 126021.05 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-06-13 12:30:00 | 126066.60 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-06-13 13:15:00 | 126072.45 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-06-13 14:45:00 | 126054.05 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-14 14:15:00 | 125605.00 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-06-14 15:00:00 | 125599.95 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-06-19 10:00:00 | 125376.05 | 2024-06-20 13:15:00 | 126320.85 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-06-25 11:30:00 | 125700.00 | 2024-06-25 13:15:00 | 126224.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-06-25 12:15:00 | 125638.95 | 2024-06-25 13:15:00 | 126224.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-07-02 14:15:00 | 129312.05 | 2024-07-03 09:15:00 | 128463.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-03 15:15:00 | 129250.00 | 2024-07-04 10:15:00 | 128500.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-07-05 09:15:00 | 129148.90 | 2024-07-05 15:15:00 | 128506.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-07-08 09:15:00 | 129658.95 | 2024-07-08 13:15:00 | 128621.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-07-08 10:30:00 | 130945.00 | 2024-07-12 13:15:00 | 130087.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-09 14:15:00 | 130950.00 | 2024-07-12 13:15:00 | 130087.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-10 09:30:00 | 131296.00 | 2024-07-12 13:15:00 | 130087.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-07-10 15:00:00 | 130900.00 | 2024-07-12 13:15:00 | 130087.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-08-06 13:30:00 | 137080.45 | 2024-08-08 13:15:00 | 141273.66 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2024-08-19 10:15:00 | 135450.05 | 2024-08-21 09:15:00 | 137350.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-19 12:00:00 | 135380.00 | 2024-08-21 09:15:00 | 137350.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-08-20 10:30:00 | 135486.05 | 2024-08-21 09:15:00 | 137350.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-20 11:30:00 | 135417.59 | 2024-08-21 09:15:00 | 137350.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest1 | 2024-08-26 09:45:00 | 139445.20 | 2024-08-26 12:15:00 | 138550.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-27 10:30:00 | 139662.09 | 2024-08-28 09:15:00 | 138500.09 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest1 | 2024-08-30 12:00:00 | 134549.00 | 2024-09-04 09:15:00 | 134999.41 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-09-02 10:15:00 | 134630.09 | 2024-09-04 14:15:00 | 135390.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-09-02 11:00:00 | 134706.20 | 2024-09-04 14:15:00 | 135390.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-09-02 11:45:00 | 134596.25 | 2024-09-04 14:15:00 | 135390.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-09-04 10:15:00 | 134819.95 | 2024-09-04 14:15:00 | 135390.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-09-06 12:15:00 | 135629.95 | 2024-09-06 12:15:00 | 135000.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-09-12 09:15:00 | 136000.30 | 2024-09-17 09:15:00 | 135550.91 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-09-12 10:15:00 | 135923.16 | 2024-09-17 09:15:00 | 135550.91 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-09-12 14:00:00 | 135999.55 | 2024-09-17 09:15:00 | 135550.91 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-09-13 09:15:00 | 136300.00 | 2024-09-17 09:15:00 | 135550.91 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-09-13 11:00:00 | 137189.16 | 2024-09-17 09:15:00 | 135550.91 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-10-10 12:45:00 | 132350.05 | 2024-10-21 11:15:00 | 130926.55 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2024-10-28 12:30:00 | 123435.00 | 2024-11-01 18:15:00 | 123379.95 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-11-18 09:15:00 | 120312.05 | 2024-11-19 10:15:00 | 121369.70 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-11-18 10:00:00 | 119977.50 | 2024-11-19 10:15:00 | 121369.70 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-11-18 13:00:00 | 120325.35 | 2024-11-19 10:15:00 | 121369.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-11-22 12:15:00 | 123437.90 | 2024-11-28 11:15:00 | 123590.00 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2024-11-27 09:45:00 | 123690.90 | 2024-11-28 11:15:00 | 123590.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-12-16 15:15:00 | 132700.00 | 2024-12-17 09:15:00 | 131822.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-12-26 09:15:00 | 130419.75 | 2024-12-30 13:15:00 | 129655.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-12-26 10:00:00 | 130457.30 | 2024-12-30 13:15:00 | 129655.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-26 11:00:00 | 130900.00 | 2024-12-30 13:15:00 | 129655.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-30 09:45:00 | 130425.85 | 2024-12-30 13:15:00 | 129655.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-01-03 10:30:00 | 128526.50 | 2025-01-07 09:15:00 | 122100.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 10:30:00 | 128526.50 | 2025-01-09 09:15:00 | 120383.85 | STOP_HIT | 0.50 | 6.34% |
| BUY | retest2 | 2025-01-30 09:30:00 | 112611.90 | 2025-02-06 13:15:00 | 113309.05 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-01-31 09:15:00 | 113000.10 | 2025-02-06 13:15:00 | 113309.05 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-02-06 13:00:00 | 113695.10 | 2025-02-06 13:15:00 | 113309.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-02-12 14:15:00 | 109196.00 | 2025-02-20 10:15:00 | 109950.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-02-14 10:45:00 | 109038.00 | 2025-02-20 10:15:00 | 109950.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-02-14 12:00:00 | 109125.55 | 2025-02-20 10:15:00 | 109950.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-02-14 12:30:00 | 109150.00 | 2025-02-20 10:15:00 | 109950.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-02-25 11:45:00 | 108288.85 | 2025-03-04 12:15:00 | 102874.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:00:00 | 108199.50 | 2025-03-04 13:15:00 | 102789.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 108288.85 | 2025-03-05 09:15:00 | 103817.50 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2025-02-25 14:00:00 | 108199.50 | 2025-03-05 09:15:00 | 103817.50 | STOP_HIT | 0.50 | 4.05% |
| BUY | retest1 | 2025-03-26 09:15:00 | 115013.00 | 2025-03-26 15:15:00 | 114008.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest1 | 2025-03-26 09:45:00 | 114656.60 | 2025-03-26 15:15:00 | 114008.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-04-01 11:15:00 | 113133.60 | 2025-04-02 10:15:00 | 114199.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-04-01 12:45:00 | 113124.00 | 2025-04-02 10:15:00 | 114199.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-04-03 15:00:00 | 114971.65 | 2025-04-04 09:15:00 | 112901.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-04-09 11:00:00 | 111450.00 | 2025-04-09 12:15:00 | 113235.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-05-07 09:15:00 | 135605.00 | 2025-05-07 09:15:00 | 134485.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-09 11:15:00 | 137630.00 | 2025-05-19 09:15:00 | 140130.00 | STOP_HIT | 1.00 | 1.82% |
| SELL | retest2 | 2025-06-03 09:30:00 | 139835.00 | 2025-06-05 13:15:00 | 139995.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-06-05 12:30:00 | 140050.00 | 2025-06-05 13:15:00 | 139995.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-06-05 13:00:00 | 140080.00 | 2025-06-05 13:15:00 | 139995.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-06-09 09:15:00 | 141010.00 | 2025-06-09 15:15:00 | 138530.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-11 12:30:00 | 137835.00 | 2025-06-18 11:15:00 | 137295.00 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest1 | 2025-06-20 09:15:00 | 138315.00 | 2025-06-20 09:15:00 | 136415.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-01 09:15:00 | 143230.00 | 2025-07-04 13:15:00 | 143690.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-07-01 13:00:00 | 142915.00 | 2025-07-04 13:15:00 | 143690.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-07-02 09:15:00 | 143215.00 | 2025-07-04 13:15:00 | 143690.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-14 09:15:00 | 149860.00 | 2025-07-18 10:15:00 | 148480.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-14 14:30:00 | 149030.00 | 2025-07-18 10:15:00 | 148480.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-15 09:15:00 | 148995.00 | 2025-07-18 10:15:00 | 148480.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-22 11:30:00 | 148855.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-22 12:45:00 | 149095.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-23 11:00:00 | 148950.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-23 12:45:00 | 148990.00 | 2025-07-23 14:15:00 | 149945.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-08-06 11:00:00 | 145725.00 | 2025-08-06 12:15:00 | 146895.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-22 13:30:00 | 147375.00 | 2025-08-25 09:15:00 | 146605.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-25 12:00:00 | 147475.00 | 2025-08-25 12:15:00 | 146955.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-12 13:45:00 | 146675.00 | 2025-09-15 09:15:00 | 148350.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-09-12 15:00:00 | 146280.00 | 2025-09-15 09:15:00 | 148350.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-19 09:15:00 | 149845.00 | 2025-09-19 09:15:00 | 148770.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-01 10:30:00 | 146330.00 | 2025-10-01 14:15:00 | 148310.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-10-01 11:15:00 | 146365.00 | 2025-10-01 14:15:00 | 148310.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-01 11:45:00 | 146430.00 | 2025-10-01 14:15:00 | 148310.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-13 10:15:00 | 156830.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-13 11:15:00 | 156395.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-13 12:15:00 | 156390.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-13 12:45:00 | 156400.00 | 2025-10-15 14:15:00 | 155185.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-10-24 09:30:00 | 162850.00 | 2025-10-24 11:15:00 | 160590.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-10-27 09:15:00 | 160620.00 | 2025-10-27 14:15:00 | 159315.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-27 11:00:00 | 160695.00 | 2025-10-27 14:15:00 | 159315.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-27 11:45:00 | 160530.00 | 2025-10-27 14:15:00 | 159315.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-29 12:45:00 | 158600.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-10-30 10:45:00 | 158500.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-10-30 11:30:00 | 158450.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-31 09:45:00 | 158255.00 | 2025-11-04 09:15:00 | 159060.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-31 11:15:00 | 157885.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-31 12:15:00 | 158030.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-03 11:30:00 | 158260.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-03 13:15:00 | 158210.00 | 2025-11-07 13:15:00 | 158580.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-11-04 11:30:00 | 157950.00 | 2025-11-07 14:15:00 | 158760.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-04 15:00:00 | 157605.00 | 2025-11-07 14:15:00 | 158760.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-06 10:15:00 | 157735.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-06 12:15:00 | 158070.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-11-06 15:00:00 | 156930.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-07 12:45:00 | 157825.00 | 2025-11-07 15:15:00 | 158605.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-08 09:15:00 | 149500.00 | 2026-01-19 11:15:00 | 142025.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:45:00 | 149370.00 | 2026-01-19 11:15:00 | 141901.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 149500.00 | 2026-01-19 14:15:00 | 143195.00 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2026-01-08 09:45:00 | 149370.00 | 2026-01-19 14:15:00 | 143195.00 | STOP_HIT | 0.50 | 4.13% |
| BUY | retest1 | 2026-02-04 09:15:00 | 139100.00 | 2026-02-05 09:15:00 | 135930.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-02-05 12:30:00 | 135675.00 | 2026-02-05 15:15:00 | 134750.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-02-13 12:00:00 | 151030.00 | 2026-02-16 11:15:00 | 147920.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-02-19 09:15:00 | 147155.00 | 2026-02-25 12:15:00 | 146580.00 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2026-03-19 09:15:00 | 132490.00 | 2026-03-23 09:15:00 | 125865.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 132490.00 | 2026-03-24 09:15:00 | 125930.00 | STOP_HIT | 0.50 | 4.95% |
| BUY | retest2 | 2026-04-13 10:15:00 | 135300.00 | 2026-04-22 10:15:00 | 137670.00 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2026-04-13 10:45:00 | 135270.00 | 2026-04-22 10:15:00 | 137670.00 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-04-15 09:15:00 | 136420.00 | 2026-04-22 10:15:00 | 137670.00 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest1 | 2026-04-28 09:30:00 | 131700.00 | 2026-04-29 09:15:00 | 133100.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-04-29 13:30:00 | 130405.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2026-04-30 09:15:00 | 129195.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-30 14:00:00 | 130645.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-04-30 14:45:00 | 130160.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-05-04 12:15:00 | 129750.00 | 2026-05-06 15:15:00 | 130300.00 | STOP_HIT | 1.00 | -0.42% |
