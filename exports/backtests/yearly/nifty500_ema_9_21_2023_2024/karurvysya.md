# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 304.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 238 |
| ALERT1 | 156 |
| ALERT2 | 149 |
| ALERT2_SKIP | 95 |
| ALERT3 | 306 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 110 |
| PARTIAL | 15 |
| TARGET_HIT | 10 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 69 / 58
- **Target hits / Stop hits / Partials:** 10 / 102 / 15
- **Avg / median % per leg:** 1.35% / 0.11%
- **Sum % (uncompounded):** 171.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 21 | 44.7% | 7 | 40 | 0 | 0.94% | 44.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.47% | -1.5% |
| BUY @ 3rd Alert (retest2) | 46 | 21 | 45.7% | 7 | 39 | 0 | 1.00% | 45.8% |
| SELL (all) | 80 | 48 | 60.0% | 3 | 62 | 15 | 1.58% | 126.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.55% | -4.7% |
| SELL @ 3rd Alert (retest2) | 77 | 48 | 62.3% | 3 | 59 | 15 | 1.71% | 131.4% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.53% | -6.1% |
| retest2 (combined) | 123 | 69 | 56.1% | 10 | 98 | 15 | 1.44% | 177.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 87.79 | 88.01 | 88.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 13:15:00 | 86.79 | 87.68 | 87.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 09:15:00 | 89.00 | 87.76 | 87.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 89.00 | 87.76 | 87.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 89.00 | 87.76 | 87.83 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 10:15:00 | 90.67 | 88.34 | 88.09 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 87.88 | 88.63 | 88.65 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 89.50 | 88.63 | 88.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 11:15:00 | 89.88 | 88.88 | 88.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 14:15:00 | 89.33 | 89.71 | 89.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 14:15:00 | 89.33 | 89.71 | 89.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 89.33 | 89.71 | 89.39 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 101.79 | 102.74 | 102.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 101.04 | 102.35 | 102.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 100.46 | 100.16 | 100.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 100.46 | 100.16 | 100.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 100.46 | 100.16 | 100.75 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 104.46 | 101.46 | 101.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 10:15:00 | 105.04 | 102.18 | 101.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 09:15:00 | 103.33 | 103.95 | 102.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 10:15:00 | 103.96 | 103.96 | 102.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 103.96 | 103.96 | 102.96 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 107.04 | 108.76 | 108.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 106.25 | 107.65 | 108.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 12:15:00 | 106.13 | 104.91 | 106.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 12:15:00 | 106.13 | 104.91 | 106.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 106.13 | 104.91 | 106.01 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 107.00 | 104.73 | 104.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 107.46 | 106.00 | 105.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 10:15:00 | 107.58 | 107.58 | 106.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 15:15:00 | 107.83 | 108.02 | 107.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 107.83 | 108.02 | 107.71 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 10:15:00 | 107.08 | 107.58 | 107.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 11:15:00 | 106.63 | 107.39 | 107.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 104.33 | 103.95 | 104.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 10:15:00 | 105.75 | 104.31 | 104.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 105.75 | 104.31 | 104.80 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 15:15:00 | 104.96 | 104.74 | 104.73 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 103.75 | 104.55 | 104.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 11:15:00 | 103.46 | 104.33 | 104.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 15:15:00 | 103.29 | 102.75 | 103.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 15:15:00 | 103.29 | 102.75 | 103.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 103.29 | 102.75 | 103.29 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 104.21 | 103.57 | 103.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 106.17 | 104.30 | 103.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 12:15:00 | 104.33 | 104.56 | 104.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 13:15:00 | 104.17 | 104.48 | 104.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 104.17 | 104.48 | 104.16 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 103.08 | 104.27 | 104.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 13:15:00 | 102.96 | 103.66 | 103.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 12:15:00 | 102.92 | 102.67 | 103.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 12:15:00 | 102.92 | 102.67 | 103.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 102.92 | 102.67 | 103.03 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 15:15:00 | 101.08 | 99.60 | 99.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 101.63 | 100.25 | 99.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 10:15:00 | 103.46 | 103.76 | 102.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 13:15:00 | 102.71 | 103.38 | 102.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 102.71 | 103.38 | 102.67 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 101.13 | 102.32 | 102.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 09:15:00 | 99.58 | 101.11 | 101.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 15:15:00 | 100.33 | 100.28 | 100.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 102.17 | 100.66 | 101.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 102.17 | 100.66 | 101.04 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 10:15:00 | 101.29 | 101.15 | 101.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 12:15:00 | 102.04 | 101.46 | 101.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 14:15:00 | 101.50 | 101.59 | 101.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 100.83 | 101.43 | 101.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 100.83 | 101.43 | 101.33 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 100.17 | 101.18 | 101.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 10:15:00 | 99.21 | 100.79 | 101.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 12:15:00 | 101.08 | 100.75 | 100.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 12:15:00 | 101.08 | 100.75 | 100.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 101.08 | 100.75 | 100.97 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 15:15:00 | 101.58 | 101.16 | 101.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 103.00 | 101.53 | 101.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 14:15:00 | 102.25 | 102.64 | 102.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 15:15:00 | 102.00 | 102.51 | 102.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 102.00 | 102.51 | 102.04 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 105.83 | 107.95 | 108.16 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 109.00 | 107.68 | 107.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 112.75 | 109.19 | 108.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 15:15:00 | 110.33 | 110.53 | 109.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 110.75 | 110.58 | 109.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 110.75 | 110.58 | 109.69 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 111.25 | 112.94 | 113.03 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 11:15:00 | 111.88 | 111.68 | 111.66 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 13:15:00 | 111.50 | 111.64 | 111.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 14:15:00 | 110.63 | 111.44 | 111.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 111.17 | 109.92 | 110.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 111.17 | 109.92 | 110.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 111.17 | 109.92 | 110.51 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 111.54 | 110.79 | 110.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 111.83 | 111.42 | 111.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 112.42 | 112.55 | 112.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 13:15:00 | 111.83 | 112.48 | 112.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 111.83 | 112.48 | 112.21 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 11:15:00 | 111.67 | 112.06 | 112.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 12:15:00 | 111.25 | 111.89 | 112.01 | Break + close below crossover candle low |

### Cycle 26 — BUY (started 2023-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 13:15:00 | 113.21 | 112.16 | 112.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 114.67 | 112.66 | 112.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 15:15:00 | 122.75 | 122.85 | 121.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 121.38 | 122.56 | 121.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 121.38 | 122.56 | 121.29 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 15:15:00 | 119.25 | 120.57 | 120.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 116.79 | 119.81 | 120.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 119.08 | 118.27 | 119.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 119.08 | 118.27 | 119.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 119.08 | 118.27 | 119.13 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 13:15:00 | 122.04 | 119.48 | 119.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 10:15:00 | 122.96 | 120.53 | 119.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-27 13:15:00 | 120.83 | 121.09 | 120.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 09:15:00 | 120.29 | 121.08 | 120.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 120.29 | 121.08 | 120.50 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 15:15:00 | 119.63 | 120.21 | 120.26 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 120.75 | 120.32 | 120.30 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 119.96 | 120.25 | 120.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 12:15:00 | 119.08 | 119.96 | 120.13 | Break + close below crossover candle low |

### Cycle 32 — BUY (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 09:15:00 | 122.04 | 120.15 | 120.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 10:15:00 | 123.79 | 120.87 | 120.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 12:15:00 | 122.88 | 123.03 | 122.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 124.67 | 124.83 | 123.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 124.67 | 124.83 | 123.98 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 15:15:00 | 123.04 | 123.62 | 123.64 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 13:15:00 | 124.88 | 123.85 | 123.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 14:15:00 | 125.83 | 124.24 | 123.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 126.00 | 126.07 | 125.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 126.79 | 126.22 | 125.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 126.79 | 126.22 | 125.42 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 11:15:00 | 129.17 | 130.00 | 130.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 13:15:00 | 128.00 | 129.68 | 129.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 10:15:00 | 126.92 | 126.83 | 127.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 13:15:00 | 127.88 | 126.98 | 127.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 13:15:00 | 127.88 | 126.98 | 127.61 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 126.92 | 126.74 | 126.73 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 10:15:00 | 125.79 | 126.55 | 126.64 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 127.08 | 126.64 | 126.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 129.33 | 127.62 | 127.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 11:15:00 | 129.46 | 129.68 | 129.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 13:15:00 | 128.50 | 129.48 | 129.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 128.50 | 129.48 | 129.03 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 09:15:00 | 129.17 | 129.66 | 129.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 10:15:00 | 128.38 | 129.40 | 129.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 130.63 | 129.24 | 129.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 130.63 | 129.24 | 129.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 130.63 | 129.24 | 129.35 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 134.63 | 130.31 | 129.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 11:15:00 | 135.67 | 131.39 | 130.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 139.04 | 140.02 | 138.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 139.92 | 140.54 | 139.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 139.92 | 140.54 | 139.52 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 138.46 | 140.12 | 140.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 137.63 | 139.62 | 140.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 136.42 | 135.65 | 137.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 136.42 | 135.65 | 137.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 136.42 | 135.65 | 137.10 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 13:15:00 | 139.04 | 137.55 | 137.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 139.58 | 137.95 | 137.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 09:15:00 | 137.04 | 137.83 | 137.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 09:15:00 | 137.04 | 137.83 | 137.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 137.04 | 137.83 | 137.69 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 13:15:00 | 137.08 | 137.57 | 137.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 14:15:00 | 136.58 | 137.37 | 137.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 10:15:00 | 138.25 | 137.21 | 137.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 10:15:00 | 138.25 | 137.21 | 137.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 138.25 | 137.21 | 137.38 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 09:15:00 | 140.17 | 137.76 | 137.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 12:15:00 | 141.25 | 139.05 | 138.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 139.46 | 139.76 | 138.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 139.46 | 139.76 | 138.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 139.46 | 139.76 | 138.91 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 09:15:00 | 138.83 | 140.09 | 140.20 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 140.83 | 140.29 | 140.27 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 13:15:00 | 139.83 | 140.26 | 140.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 15:15:00 | 139.42 | 140.01 | 140.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 140.67 | 140.14 | 140.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 140.67 | 140.14 | 140.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 140.67 | 140.14 | 140.19 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 10:15:00 | 138.08 | 137.62 | 137.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 11:15:00 | 138.83 | 137.86 | 137.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 143.42 | 143.98 | 142.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 142.50 | 143.75 | 142.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 142.50 | 143.75 | 142.91 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 14:15:00 | 141.08 | 142.51 | 142.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 15:15:00 | 140.92 | 142.19 | 142.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 142.92 | 142.20 | 142.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 142.92 | 142.20 | 142.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 142.92 | 142.20 | 142.37 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 11:15:00 | 144.38 | 142.63 | 142.55 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 12:15:00 | 141.79 | 142.52 | 142.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-19 13:15:00 | 140.63 | 142.14 | 142.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 12:15:00 | 141.79 | 141.71 | 142.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 12:15:00 | 141.79 | 141.71 | 142.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 141.79 | 141.71 | 142.02 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 151.50 | 143.66 | 142.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-23 11:15:00 | 154.96 | 147.10 | 144.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 15:15:00 | 149.83 | 150.42 | 147.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 15:15:00 | 149.71 | 150.88 | 149.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 149.71 | 150.88 | 149.28 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 161.00 | 162.94 | 162.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 158.75 | 160.88 | 161.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 156.50 | 154.93 | 156.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 156.50 | 154.93 | 156.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 156.50 | 154.93 | 156.40 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 152.83 | 149.81 | 149.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 154.21 | 151.70 | 150.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 12:15:00 | 152.29 | 152.29 | 151.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 10:15:00 | 153.54 | 155.96 | 154.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 153.54 | 155.96 | 154.78 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 15:15:00 | 152.71 | 154.03 | 154.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 151.58 | 153.54 | 153.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 14:15:00 | 152.50 | 152.23 | 153.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 14:15:00 | 152.50 | 152.23 | 153.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 152.50 | 152.23 | 153.01 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 153.29 | 150.77 | 150.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 09:15:00 | 155.83 | 153.50 | 152.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 15:15:00 | 154.50 | 154.57 | 153.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 09:15:00 | 151.67 | 153.99 | 153.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 151.67 | 153.99 | 153.25 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 152.58 | 154.55 | 154.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 149.92 | 153.03 | 153.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 09:15:00 | 149.17 | 149.07 | 150.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 10:15:00 | 147.00 | 148.66 | 150.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 147.00 | 148.66 | 150.03 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-03-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 12:15:00 | 148.58 | 146.26 | 146.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 150.17 | 148.08 | 147.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 147.33 | 148.29 | 147.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 147.33 | 148.29 | 147.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 147.33 | 148.29 | 147.66 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 145.58 | 147.12 | 147.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 10:15:00 | 144.25 | 146.55 | 147.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 145.79 | 145.56 | 146.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 10:15:00 | 145.79 | 145.56 | 146.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 145.79 | 145.56 | 146.17 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 147.25 | 146.34 | 146.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 149.71 | 147.01 | 146.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 150.04 | 150.17 | 148.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 150.04 | 150.17 | 148.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 150.04 | 150.17 | 148.72 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 157.75 | 159.51 | 159.63 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 160.08 | 159.70 | 159.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 160.42 | 159.85 | 159.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 10:15:00 | 160.00 | 160.22 | 160.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 10:15:00 | 160.00 | 160.22 | 160.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 160.00 | 160.22 | 160.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 11:00:00 | 160.00 | 160.22 | 160.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 160.29 | 160.23 | 160.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 11:30:00 | 160.38 | 160.23 | 160.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 159.83 | 160.15 | 160.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:45:00 | 159.58 | 160.15 | 160.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 159.75 | 160.07 | 159.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:00:00 | 159.75 | 160.07 | 159.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 159.92 | 160.04 | 159.98 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 158.13 | 159.66 | 159.82 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 160.21 | 158.65 | 158.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 12:15:00 | 160.88 | 159.10 | 158.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 158.63 | 159.00 | 158.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 13:15:00 | 158.63 | 159.00 | 158.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 158.63 | 159.00 | 158.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:00:00 | 158.63 | 159.00 | 158.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 157.04 | 158.61 | 158.61 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 159.00 | 158.60 | 158.58 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 14:15:00 | 158.17 | 158.51 | 158.54 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 160.08 | 158.76 | 158.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 161.54 | 159.83 | 159.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 160.42 | 160.51 | 159.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 11:00:00 | 160.42 | 160.51 | 159.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 159.88 | 160.31 | 159.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:45:00 | 159.83 | 160.31 | 159.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 160.08 | 160.27 | 159.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:30:00 | 159.88 | 160.27 | 159.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 159.88 | 160.19 | 159.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:45:00 | 160.38 | 160.19 | 159.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 160.00 | 160.15 | 159.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:30:00 | 160.13 | 160.00 | 159.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 159.17 | 159.83 | 159.73 | SL hit (close<static) qty=1.00 sl=159.25 alert=retest2 |

### Cycle 69 — SELL (started 2024-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 12:15:00 | 159.00 | 159.61 | 159.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 158.67 | 159.34 | 159.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 09:15:00 | 160.46 | 159.27 | 159.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 160.46 | 159.27 | 159.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 160.46 | 159.27 | 159.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 160.46 | 159.27 | 159.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 160.33 | 159.48 | 159.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 11:30:00 | 159.17 | 159.23 | 159.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 14:45:00 | 159.42 | 159.01 | 159.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 160.54 | 159.39 | 159.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 160.54 | 159.39 | 159.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 09:15:00 | 161.58 | 159.89 | 159.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 11:15:00 | 169.75 | 169.79 | 167.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 11:45:00 | 170.21 | 169.79 | 167.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 169.08 | 169.77 | 168.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 166.58 | 169.77 | 168.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 164.46 | 168.71 | 168.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:30:00 | 164.75 | 168.71 | 168.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 164.17 | 167.80 | 167.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:15:00 | 164.38 | 167.80 | 167.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 11:15:00 | 162.88 | 166.82 | 167.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 12:15:00 | 162.29 | 165.91 | 166.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 158.08 | 157.84 | 160.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 158.08 | 157.84 | 160.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 159.21 | 157.62 | 158.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 159.21 | 157.62 | 158.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 157.04 | 157.50 | 158.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 15:15:00 | 155.83 | 157.50 | 158.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:45:00 | 156.67 | 157.18 | 157.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 160.38 | 156.75 | 156.95 | SL hit (close>static) qty=1.00 sl=159.42 alert=retest2 |

### Cycle 72 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 160.54 | 157.50 | 157.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 161.63 | 159.55 | 158.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 164.42 | 164.96 | 163.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:45:00 | 164.75 | 164.96 | 163.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 164.79 | 165.30 | 164.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 164.21 | 165.08 | 164.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 164.83 | 165.03 | 164.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 165.08 | 165.03 | 164.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 165.33 | 165.09 | 164.56 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 163.33 | 164.30 | 164.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 162.50 | 163.94 | 164.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 163.71 | 163.70 | 164.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 163.71 | 163.70 | 164.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 163.71 | 163.70 | 164.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 163.96 | 163.70 | 164.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 163.33 | 163.63 | 163.95 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 164.54 | 164.08 | 164.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 166.00 | 164.46 | 164.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 15:15:00 | 165.33 | 165.35 | 164.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 15:15:00 | 165.33 | 165.35 | 164.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 165.33 | 165.35 | 164.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 165.71 | 165.35 | 164.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 11:15:00 | 164.04 | 165.00 | 164.84 | SL hit (close<static) qty=1.00 sl=164.75 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 163.92 | 164.63 | 164.69 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 166.96 | 164.96 | 164.80 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 11:15:00 | 164.21 | 164.95 | 165.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 12:15:00 | 163.83 | 164.73 | 164.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 14:15:00 | 163.79 | 163.21 | 163.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 14:15:00 | 163.79 | 163.21 | 163.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 163.79 | 163.21 | 163.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:00:00 | 161.54 | 162.90 | 163.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 162.21 | 162.88 | 163.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 165.54 | 163.64 | 163.67 | SL hit (close>static) qty=1.00 sl=164.58 alert=retest2 |

### Cycle 78 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 166.21 | 163.97 | 163.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 167.33 | 165.30 | 164.51 | Break + close above crossover candle high |

### Cycle 79 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 159.63 | 164.35 | 164.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 153.29 | 162.13 | 163.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 155.83 | 153.83 | 157.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 155.83 | 153.83 | 157.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 159.54 | 154.97 | 157.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:45:00 | 158.75 | 154.97 | 157.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 160.17 | 156.01 | 157.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 160.17 | 156.01 | 157.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 165.96 | 160.12 | 159.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 166.33 | 162.31 | 160.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 11:15:00 | 175.67 | 176.14 | 172.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:00:00 | 175.67 | 176.14 | 172.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 173.21 | 175.69 | 174.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 173.21 | 175.69 | 174.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 173.92 | 175.33 | 174.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:30:00 | 172.54 | 174.79 | 174.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 172.75 | 174.38 | 174.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:45:00 | 172.69 | 174.38 | 174.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 12:15:00 | 172.87 | 173.85 | 173.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 14:15:00 | 172.03 | 173.32 | 173.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 10:15:00 | 173.21 | 173.08 | 173.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 10:15:00 | 173.21 | 173.08 | 173.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 173.21 | 173.08 | 173.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:30:00 | 173.33 | 173.08 | 173.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 173.25 | 173.12 | 173.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:45:00 | 173.21 | 173.12 | 173.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 173.38 | 173.17 | 173.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:30:00 | 173.42 | 173.17 | 173.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 173.00 | 173.14 | 173.38 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 175.87 | 173.89 | 173.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 178.83 | 177.38 | 175.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 11:15:00 | 178.08 | 178.65 | 177.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 178.08 | 178.65 | 177.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 178.08 | 178.65 | 177.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 178.08 | 178.65 | 177.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 177.54 | 178.43 | 177.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:45:00 | 177.74 | 178.43 | 177.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 177.08 | 178.16 | 177.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:45:00 | 176.79 | 178.16 | 177.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 177.32 | 177.99 | 177.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:30:00 | 177.33 | 177.99 | 177.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 174.79 | 177.23 | 177.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 174.58 | 175.79 | 176.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 15:15:00 | 176.25 | 175.28 | 175.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 15:15:00 | 176.25 | 175.28 | 175.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 176.25 | 175.28 | 175.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 175.98 | 175.28 | 175.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 174.50 | 175.12 | 175.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:30:00 | 173.58 | 174.83 | 175.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 172.00 | 170.82 | 170.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 172.00 | 170.82 | 170.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 174.00 | 171.97 | 171.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 172.13 | 172.69 | 172.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 15:15:00 | 172.13 | 172.69 | 172.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 172.13 | 172.69 | 172.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 171.68 | 172.52 | 172.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 171.67 | 172.35 | 171.97 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 170.08 | 171.61 | 171.68 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 172.83 | 171.82 | 171.69 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 171.10 | 171.64 | 171.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 169.70 | 171.16 | 171.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 11:15:00 | 161.63 | 160.77 | 162.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 12:00:00 | 161.63 | 160.77 | 162.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 163.43 | 161.02 | 162.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 163.43 | 161.02 | 162.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 163.13 | 161.44 | 162.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 163.57 | 161.44 | 162.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 164.54 | 162.40 | 162.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 164.54 | 162.40 | 162.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 166.28 | 163.18 | 163.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 12:15:00 | 166.79 | 163.90 | 163.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 164.93 | 165.08 | 164.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 164.93 | 165.08 | 164.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 164.93 | 165.08 | 164.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 163.69 | 165.08 | 164.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 164.63 | 164.99 | 164.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:15:00 | 163.88 | 164.99 | 164.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 165.72 | 165.14 | 164.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:30:00 | 165.84 | 165.45 | 164.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-26 09:15:00 | 182.42 | 179.34 | 177.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 187.58 | 189.26 | 189.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 184.48 | 188.04 | 188.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 178.46 | 177.71 | 180.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 178.46 | 177.71 | 180.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 177.43 | 175.82 | 177.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 177.71 | 175.82 | 177.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 180.17 | 176.69 | 177.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 180.17 | 176.69 | 177.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 178.61 | 177.08 | 177.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 178.81 | 177.08 | 177.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 177.92 | 177.29 | 177.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 177.92 | 177.29 | 177.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 177.00 | 177.23 | 177.71 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 179.13 | 177.82 | 177.69 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 176.48 | 177.77 | 177.89 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 180.83 | 178.39 | 178.16 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 177.85 | 178.24 | 178.25 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 178.42 | 178.28 | 178.26 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 177.93 | 178.25 | 178.25 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 178.75 | 178.30 | 178.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 181.02 | 179.00 | 178.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 182.66 | 183.15 | 181.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 182.66 | 183.15 | 181.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 186.25 | 187.82 | 186.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 186.25 | 187.82 | 186.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 186.65 | 187.59 | 186.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 187.08 | 187.59 | 186.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 185.38 | 187.14 | 186.33 | SL hit (close<static) qty=1.00 sl=185.68 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 185.59 | 185.93 | 185.95 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 187.10 | 186.11 | 186.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 187.61 | 186.77 | 186.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 187.43 | 187.69 | 187.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 187.43 | 187.69 | 187.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 187.43 | 187.69 | 187.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 187.43 | 187.69 | 187.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 186.88 | 187.52 | 187.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 186.88 | 187.52 | 187.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 186.71 | 187.36 | 187.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:15:00 | 187.08 | 187.36 | 187.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 186.78 | 187.25 | 187.01 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 185.50 | 186.82 | 186.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 184.16 | 185.94 | 186.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 13:15:00 | 185.99 | 185.93 | 186.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 13:15:00 | 185.99 | 185.93 | 186.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 185.99 | 185.93 | 186.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:30:00 | 186.02 | 185.93 | 186.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 186.00 | 185.94 | 186.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 186.00 | 185.94 | 186.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 187.38 | 186.25 | 186.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 186.38 | 186.25 | 186.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 186.85 | 186.37 | 186.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 186.73 | 186.42 | 186.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 12:15:00 | 187.12 | 186.56 | 186.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 187.12 | 186.56 | 186.49 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 184.75 | 186.18 | 186.33 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 15:15:00 | 186.63 | 186.31 | 186.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 187.71 | 186.59 | 186.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 186.67 | 186.79 | 186.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 13:15:00 | 186.67 | 186.79 | 186.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 186.67 | 186.79 | 186.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:45:00 | 186.46 | 186.79 | 186.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 187.49 | 186.93 | 186.67 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 183.87 | 186.01 | 186.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 11:15:00 | 182.63 | 185.34 | 185.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 185.07 | 184.20 | 185.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 185.07 | 184.20 | 185.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 185.07 | 184.20 | 185.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 185.17 | 184.20 | 185.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 185.79 | 184.52 | 185.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:30:00 | 185.60 | 184.52 | 185.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 185.00 | 184.62 | 185.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:00:00 | 184.42 | 184.70 | 185.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 186.17 | 183.17 | 182.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 186.17 | 183.17 | 182.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 187.05 | 183.95 | 183.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 183.20 | 184.24 | 183.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 183.20 | 184.24 | 183.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 183.20 | 184.24 | 183.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 183.20 | 184.24 | 183.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 184.23 | 184.24 | 183.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 13:15:00 | 184.48 | 184.22 | 183.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 182.48 | 183.85 | 183.59 | SL hit (close<static) qty=1.00 sl=183.20 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 183.08 | 183.36 | 183.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 10:15:00 | 181.65 | 183.02 | 183.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 182.83 | 182.47 | 182.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 182.83 | 182.47 | 182.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 182.83 | 182.47 | 182.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:30:00 | 183.34 | 182.47 | 182.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 183.25 | 182.63 | 182.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 182.47 | 182.63 | 182.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 181.76 | 182.46 | 182.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 10:15:00 | 181.65 | 182.46 | 182.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 172.57 | 175.83 | 177.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-19 13:15:00 | 176.29 | 175.26 | 176.84 | SL hit (close>ema200) qty=0.50 sl=175.26 alert=retest2 |

### Cycle 106 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 180.42 | 176.97 | 176.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 182.38 | 178.05 | 177.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 182.92 | 183.09 | 181.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:30:00 | 182.48 | 183.09 | 181.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 182.63 | 183.33 | 182.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 182.46 | 183.33 | 182.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 181.67 | 182.86 | 182.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:30:00 | 181.61 | 182.86 | 182.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 181.62 | 182.61 | 182.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:45:00 | 181.16 | 182.61 | 182.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 180.83 | 181.97 | 182.02 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 183.50 | 182.27 | 182.15 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 181.25 | 182.21 | 182.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 179.69 | 181.71 | 182.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 10:15:00 | 179.23 | 179.19 | 180.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 11:00:00 | 179.23 | 179.19 | 180.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 168.16 | 173.12 | 175.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 167.08 | 173.12 | 175.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:30:00 | 167.80 | 171.08 | 173.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 167.24 | 165.86 | 165.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 167.24 | 165.86 | 165.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 169.13 | 166.52 | 166.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 10:15:00 | 167.51 | 167.94 | 167.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 11:00:00 | 167.51 | 167.94 | 167.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 167.12 | 167.78 | 167.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:30:00 | 167.01 | 167.78 | 167.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 166.85 | 167.59 | 167.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:30:00 | 167.04 | 167.59 | 167.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 168.02 | 167.68 | 167.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:30:00 | 169.42 | 167.98 | 167.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:00:00 | 168.36 | 168.25 | 167.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:45:00 | 168.40 | 168.41 | 167.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 11:30:00 | 168.65 | 168.48 | 168.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 167.52 | 168.29 | 168.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 13:45:00 | 171.48 | 169.12 | 168.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-18 15:15:00 | 186.36 | 180.54 | 175.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 178.97 | 180.10 | 180.19 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 14:15:00 | 183.04 | 180.28 | 180.15 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 12:15:00 | 179.08 | 180.00 | 180.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 15:15:00 | 178.25 | 179.21 | 179.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 180.42 | 179.46 | 179.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 180.42 | 179.46 | 179.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 180.42 | 179.46 | 179.74 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 181.98 | 180.29 | 180.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 182.01 | 180.84 | 180.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 186.89 | 187.63 | 185.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 09:30:00 | 186.77 | 187.63 | 185.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 187.07 | 188.02 | 187.02 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 185.78 | 186.54 | 186.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 185.24 | 186.14 | 186.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 187.14 | 186.18 | 186.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 187.14 | 186.18 | 186.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 187.14 | 186.18 | 186.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 187.77 | 186.18 | 186.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 187.42 | 186.43 | 186.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 187.42 | 186.43 | 186.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 186.67 | 186.48 | 186.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 188.17 | 186.98 | 186.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 189.01 | 191.78 | 190.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 189.01 | 191.78 | 190.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 189.01 | 191.78 | 190.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 189.05 | 191.78 | 190.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 188.01 | 191.03 | 190.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 188.01 | 191.03 | 190.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 190.08 | 190.06 | 189.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 191.13 | 190.06 | 189.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 189.96 | 190.04 | 189.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 189.64 | 190.04 | 189.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 188.80 | 189.79 | 189.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:30:00 | 191.22 | 190.20 | 189.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 14:15:00 | 189.19 | 189.90 | 189.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 189.19 | 189.90 | 189.92 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 191.63 | 190.11 | 190.00 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 188.96 | 189.88 | 189.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 186.94 | 188.63 | 189.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 13:15:00 | 178.21 | 178.10 | 180.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 14:15:00 | 176.89 | 178.10 | 180.96 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 174.68 | 176.78 | 179.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 179.45 | 176.96 | 178.72 | SL hit (close>ema400) qty=1.00 sl=178.72 alert=retest1 |

### Cycle 120 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 181.37 | 179.34 | 179.29 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 178.08 | 179.38 | 179.49 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 179.99 | 179.38 | 179.34 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 176.82 | 178.90 | 179.13 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 185.43 | 179.93 | 179.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 15:15:00 | 186.17 | 183.99 | 182.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 188.39 | 188.59 | 185.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 188.39 | 188.59 | 185.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 196.53 | 197.51 | 195.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 200.16 | 197.57 | 196.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 12:00:00 | 199.22 | 198.00 | 196.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:15:00 | 199.06 | 198.19 | 197.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 199.85 | 198.42 | 197.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 200.10 | 198.76 | 197.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:30:00 | 201.66 | 200.34 | 199.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:00:00 | 203.08 | 200.34 | 199.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 203.75 | 200.50 | 199.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 14:15:00 | 202.49 | 200.79 | 200.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 201.06 | 201.05 | 200.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 201.31 | 201.05 | 200.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 202.45 | 201.82 | 201.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 202.45 | 201.82 | 201.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 201.67 | 201.79 | 201.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 201.04 | 201.79 | 201.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 200.25 | 201.48 | 201.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 200.25 | 201.48 | 201.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 200.59 | 201.30 | 201.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:45:00 | 200.00 | 201.30 | 201.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 203.09 | 201.84 | 201.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 201.55 | 201.84 | 201.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 202.08 | 202.17 | 201.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 201.40 | 201.99 | 201.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 201.24 | 201.84 | 201.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:30:00 | 201.35 | 201.84 | 201.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 200.55 | 201.58 | 201.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:00:00 | 200.55 | 201.58 | 201.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-11 12:15:00 | 199.98 | 201.26 | 201.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 12:15:00 | 199.98 | 201.26 | 201.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 199.43 | 200.76 | 201.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 14:15:00 | 198.46 | 198.41 | 199.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:15:00 | 195.94 | 198.43 | 199.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 194.73 | 197.69 | 198.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 194.30 | 197.69 | 198.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 14:15:00 | 198.79 | 197.16 | 198.09 | SL hit (close>ema400) qty=1.00 sl=198.09 alert=retest1 |

### Cycle 126 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 186.03 | 182.82 | 182.52 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 180.58 | 182.66 | 182.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 15:15:00 | 175.38 | 181.21 | 182.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 181.73 | 180.79 | 181.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 181.73 | 180.79 | 181.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 181.73 | 180.79 | 181.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:45:00 | 181.96 | 180.79 | 181.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 180.90 | 180.81 | 181.39 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 186.25 | 182.03 | 181.77 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 183.85 | 184.93 | 184.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 182.16 | 184.38 | 184.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 182.03 | 181.75 | 182.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 182.03 | 181.75 | 182.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 179.25 | 181.25 | 182.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 178.17 | 180.53 | 181.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 178.64 | 179.28 | 180.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:45:00 | 177.76 | 178.89 | 180.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 12:15:00 | 179.78 | 176.20 | 175.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 12:15:00 | 179.78 | 176.20 | 175.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 13:15:00 | 180.23 | 177.00 | 176.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 12:15:00 | 178.38 | 178.42 | 177.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 13:00:00 | 178.38 | 178.42 | 177.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 177.02 | 178.14 | 177.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 177.02 | 178.14 | 177.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 177.10 | 177.93 | 177.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:30:00 | 177.72 | 177.93 | 177.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 177.08 | 177.76 | 177.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 177.85 | 177.76 | 177.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 185.57 | 188.22 | 188.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 185.57 | 188.22 | 188.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 184.57 | 187.49 | 187.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 185.93 | 185.30 | 186.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 185.93 | 185.30 | 186.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 185.93 | 185.30 | 186.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 185.93 | 185.30 | 186.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 187.01 | 185.64 | 186.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 187.01 | 185.64 | 186.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 187.06 | 185.92 | 186.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:15:00 | 187.63 | 185.92 | 186.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 186.93 | 186.12 | 186.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 187.35 | 186.12 | 186.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 189.08 | 186.72 | 186.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 189.33 | 186.72 | 186.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 189.83 | 187.34 | 187.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 191.50 | 189.71 | 188.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 196.71 | 197.33 | 195.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 196.71 | 197.33 | 195.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 196.71 | 197.33 | 195.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 195.92 | 197.33 | 195.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 195.08 | 196.88 | 195.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 195.08 | 196.88 | 195.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 198.23 | 197.15 | 195.47 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 193.41 | 194.85 | 194.96 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 196.88 | 194.92 | 194.91 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 15:15:00 | 194.59 | 194.97 | 194.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 09:15:00 | 194.24 | 194.82 | 194.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 09:15:00 | 194.58 | 193.96 | 194.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 194.58 | 193.96 | 194.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 194.58 | 193.96 | 194.32 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 13:15:00 | 194.79 | 194.58 | 194.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 196.17 | 195.02 | 194.77 | Break + close above crossover candle high |

### Cycle 137 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 192.71 | 194.56 | 194.58 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 197.31 | 194.61 | 194.52 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 192.11 | 194.06 | 194.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 189.64 | 191.66 | 192.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 191.25 | 190.31 | 191.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 09:15:00 | 187.62 | 190.31 | 191.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 185.76 | 189.40 | 191.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:00:00 | 185.10 | 187.58 | 188.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 184.66 | 187.00 | 188.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 184.42 | 185.83 | 187.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:00:00 | 184.36 | 185.54 | 186.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 180.48 | 181.32 | 182.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:00:00 | 180.00 | 181.02 | 182.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:45:00 | 180.15 | 180.56 | 181.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:00:00 | 179.43 | 181.10 | 181.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 175.84 | 177.78 | 179.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 175.43 | 177.78 | 179.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 175.20 | 177.78 | 179.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 175.14 | 177.78 | 179.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 176.04 | 176.02 | 177.42 | SL hit (close>ema200) qty=0.50 sl=176.02 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 12:15:00 | 168.63 | 166.47 | 166.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 13:15:00 | 171.11 | 167.40 | 166.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 169.48 | 169.74 | 168.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 169.48 | 169.74 | 168.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 167.92 | 169.33 | 168.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 168.48 | 169.33 | 168.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 166.40 | 168.74 | 168.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 166.40 | 168.74 | 168.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 166.04 | 168.20 | 168.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 165.06 | 167.57 | 168.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 10:15:00 | 158.26 | 157.87 | 160.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 11:00:00 | 158.26 | 157.87 | 160.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 161.74 | 158.64 | 160.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 161.74 | 158.64 | 160.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 163.26 | 159.57 | 160.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 163.26 | 159.57 | 160.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 163.83 | 161.53 | 161.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 15:15:00 | 168.65 | 167.08 | 165.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 15:15:00 | 174.69 | 175.83 | 174.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 09:15:00 | 174.08 | 175.83 | 174.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 174.96 | 175.66 | 174.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 175.25 | 175.66 | 174.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 174.16 | 175.33 | 174.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 174.16 | 175.33 | 174.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 174.63 | 175.19 | 174.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 13:15:00 | 175.07 | 175.19 | 174.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:00:00 | 175.09 | 175.02 | 174.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 175.08 | 175.17 | 174.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 173.21 | 174.85 | 174.82 | SL hit (close<static) qty=1.00 sl=173.51 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 174.18 | 174.81 | 174.85 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 177.42 | 175.23 | 175.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 179.25 | 178.17 | 177.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 178.71 | 179.08 | 178.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 13:00:00 | 178.71 | 179.08 | 178.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 145 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 167.56 | 177.41 | 177.86 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 176.41 | 175.14 | 175.11 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 173.08 | 174.73 | 174.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 170.50 | 173.88 | 174.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 173.56 | 172.03 | 173.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 173.56 | 172.03 | 173.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 173.56 | 172.03 | 173.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:00:00 | 173.56 | 172.03 | 173.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 173.03 | 172.23 | 173.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:30:00 | 173.16 | 172.23 | 173.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 173.33 | 172.45 | 173.09 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 174.88 | 173.60 | 173.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 176.83 | 174.24 | 173.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 187.92 | 188.15 | 186.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:15:00 | 187.39 | 188.15 | 186.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 183.35 | 187.19 | 185.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 183.35 | 187.19 | 185.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 184.24 | 186.60 | 185.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 181.80 | 186.60 | 185.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 184.99 | 185.42 | 185.42 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 186.02 | 185.40 | 185.39 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 178.88 | 184.22 | 184.89 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 11:15:00 | 183.37 | 182.35 | 182.21 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 181.25 | 181.99 | 182.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 180.17 | 181.69 | 181.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 13:15:00 | 178.28 | 178.22 | 179.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 176.70 | 177.78 | 179.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 176.70 | 177.78 | 179.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 175.83 | 177.39 | 178.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:45:00 | 174.68 | 176.32 | 177.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:45:00 | 176.03 | 175.90 | 177.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 176.01 | 176.14 | 177.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 177.48 | 176.47 | 177.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 177.48 | 176.47 | 177.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 177.50 | 176.68 | 177.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 178.17 | 176.68 | 177.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 178.22 | 176.99 | 177.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 175.96 | 177.05 | 177.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 176.53 | 176.96 | 177.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:00:00 | 176.93 | 174.81 | 175.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:45:00 | 176.97 | 175.47 | 175.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 175.83 | 175.66 | 175.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 175.83 | 175.66 | 175.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 178.70 | 176.37 | 175.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 183.13 | 183.57 | 181.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 183.46 | 183.57 | 181.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 189.50 | 190.12 | 188.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 189.50 | 190.12 | 188.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 188.87 | 189.87 | 188.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 188.87 | 189.87 | 188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 188.32 | 189.56 | 188.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 188.33 | 189.56 | 188.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 188.96 | 189.44 | 188.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 189.37 | 189.44 | 188.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 187.03 | 188.67 | 188.63 | SL hit (close<static) qty=1.00 sl=188.13 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 185.73 | 188.08 | 188.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 185.10 | 187.49 | 188.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 188.10 | 186.23 | 186.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 188.10 | 186.23 | 186.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 188.10 | 186.23 | 186.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 188.10 | 186.23 | 186.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 188.23 | 186.63 | 187.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 187.92 | 186.63 | 187.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 188.52 | 187.56 | 187.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 191.18 | 188.43 | 187.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 190.29 | 190.49 | 189.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 188.75 | 190.49 | 189.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 188.20 | 190.03 | 189.26 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 186.83 | 188.77 | 188.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 185.17 | 188.05 | 188.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 184.78 | 184.61 | 185.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 15:00:00 | 184.78 | 184.61 | 185.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 183.11 | 184.29 | 185.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 182.67 | 184.29 | 185.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 182.25 | 184.01 | 185.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 182.57 | 183.76 | 185.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 192.42 | 185.93 | 185.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 192.42 | 185.93 | 185.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 194.60 | 187.66 | 186.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 196.87 | 197.02 | 193.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 196.87 | 197.02 | 193.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 192.77 | 195.81 | 193.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 192.64 | 195.81 | 193.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 193.51 | 195.35 | 193.88 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 191.42 | 193.18 | 193.26 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 194.71 | 193.26 | 193.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 198.37 | 195.04 | 194.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 196.10 | 196.16 | 194.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 196.10 | 196.16 | 194.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 197.90 | 197.68 | 196.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:45:00 | 201.30 | 198.02 | 197.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 203.62 | 205.28 | 205.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 203.62 | 205.28 | 205.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 202.29 | 204.68 | 205.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 205.32 | 204.11 | 204.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 205.32 | 204.11 | 204.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 205.32 | 204.11 | 204.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:15:00 | 206.28 | 204.11 | 204.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 206.83 | 204.65 | 204.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 206.54 | 204.65 | 204.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 206.73 | 205.07 | 205.03 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 204.17 | 205.10 | 205.21 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 206.89 | 205.46 | 205.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 209.36 | 206.24 | 205.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 206.93 | 207.63 | 206.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 11:15:00 | 206.93 | 207.63 | 206.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 206.93 | 207.63 | 206.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 206.93 | 207.63 | 206.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 208.08 | 207.72 | 207.03 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 205.54 | 206.64 | 206.73 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 208.71 | 207.06 | 206.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 209.34 | 208.27 | 207.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 224.17 | 225.96 | 221.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 224.17 | 225.96 | 221.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 223.75 | 225.34 | 223.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 223.75 | 225.34 | 223.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 222.58 | 224.78 | 223.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 222.58 | 224.78 | 223.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 220.50 | 223.93 | 223.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:45:00 | 220.83 | 223.93 | 223.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 220.21 | 222.72 | 222.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 219.75 | 221.48 | 222.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 11:15:00 | 225.29 | 222.25 | 222.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 225.29 | 222.25 | 222.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 225.29 | 222.25 | 222.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 225.29 | 222.25 | 222.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 225.71 | 222.94 | 222.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 227.25 | 225.61 | 224.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 226.33 | 226.77 | 226.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 226.33 | 226.77 | 226.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 226.33 | 226.77 | 226.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 226.04 | 226.77 | 226.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 226.58 | 226.72 | 226.15 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 224.04 | 225.73 | 225.81 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 226.58 | 225.93 | 225.88 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 224.58 | 225.66 | 225.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 223.96 | 225.21 | 225.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 15:15:00 | 224.21 | 224.15 | 224.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 09:15:00 | 222.75 | 224.15 | 224.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 224.08 | 224.14 | 224.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 223.88 | 224.14 | 224.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 224.63 | 224.24 | 224.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 224.63 | 224.24 | 224.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 224.67 | 224.32 | 224.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 226.67 | 224.83 | 224.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 172 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 226.67 | 224.83 | 224.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 228.13 | 226.22 | 225.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 226.25 | 226.50 | 225.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 226.25 | 226.50 | 225.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 226.25 | 226.50 | 225.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 226.25 | 226.50 | 225.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 228.21 | 226.77 | 226.03 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 225.21 | 226.48 | 226.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 224.54 | 225.87 | 226.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 229.13 | 225.50 | 225.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 229.13 | 225.50 | 225.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 229.13 | 225.50 | 225.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 229.13 | 225.50 | 225.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 227.08 | 225.82 | 225.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 225.50 | 225.65 | 225.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 226.29 | 225.77 | 225.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 226.29 | 225.77 | 225.74 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 225.29 | 225.67 | 225.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 224.17 | 225.37 | 225.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 226.42 | 225.58 | 225.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 226.42 | 225.58 | 225.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 226.42 | 225.58 | 225.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 226.67 | 225.58 | 225.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 226.21 | 225.71 | 225.70 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 224.54 | 225.56 | 225.64 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 226.42 | 225.82 | 225.75 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 224.50 | 225.56 | 225.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 223.54 | 224.66 | 225.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 13:15:00 | 225.17 | 224.76 | 225.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 13:15:00 | 225.17 | 224.76 | 225.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 225.17 | 224.76 | 225.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 225.17 | 224.76 | 225.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 225.38 | 224.88 | 225.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 225.42 | 224.88 | 225.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 225.08 | 224.92 | 225.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 222.92 | 224.92 | 225.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 218.38 | 217.03 | 217.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 218.38 | 217.03 | 217.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 15:15:00 | 220.08 | 218.10 | 217.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 217.38 | 217.95 | 217.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 217.38 | 217.95 | 217.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 217.38 | 217.95 | 217.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 217.38 | 217.95 | 217.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 216.54 | 217.67 | 217.45 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 216.42 | 217.25 | 217.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 215.13 | 216.70 | 217.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 216.75 | 216.71 | 216.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 216.75 | 216.71 | 216.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 216.75 | 216.71 | 216.99 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 219.58 | 217.45 | 217.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 220.21 | 218.00 | 217.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 218.38 | 218.42 | 217.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:15:00 | 217.63 | 218.42 | 217.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 217.67 | 218.27 | 217.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 217.96 | 218.27 | 217.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 217.33 | 218.08 | 217.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 217.33 | 218.08 | 217.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 217.17 | 217.90 | 217.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 217.17 | 217.90 | 217.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 216.25 | 217.57 | 217.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 214.25 | 216.67 | 217.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 216.79 | 216.37 | 216.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 12:15:00 | 216.79 | 216.37 | 216.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 216.79 | 216.37 | 216.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 216.79 | 216.37 | 216.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 216.92 | 216.48 | 216.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 217.25 | 216.48 | 216.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 218.17 | 216.82 | 216.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 218.25 | 216.82 | 216.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 218.33 | 217.12 | 217.11 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 215.17 | 216.73 | 216.94 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 218.88 | 217.13 | 216.91 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 217.21 | 217.74 | 217.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 216.25 | 217.41 | 217.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 14:15:00 | 214.29 | 214.03 | 214.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 214.29 | 214.03 | 214.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 218.54 | 214.98 | 215.15 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 218.96 | 215.78 | 215.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 219.83 | 216.59 | 215.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 11:15:00 | 218.63 | 218.85 | 217.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:00:00 | 218.63 | 218.85 | 217.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 218.54 | 218.79 | 217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 218.08 | 218.79 | 217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 216.38 | 218.38 | 217.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:15:00 | 215.75 | 218.38 | 217.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 215.42 | 217.79 | 217.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 215.42 | 217.79 | 217.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 215.58 | 217.35 | 217.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 214.58 | 216.06 | 216.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 216.13 | 216.08 | 216.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:00:00 | 216.13 | 216.08 | 216.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 217.42 | 216.02 | 216.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 217.42 | 216.02 | 216.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 216.71 | 216.16 | 216.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 216.25 | 216.16 | 216.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 216.54 | 216.20 | 216.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 219.21 | 216.65 | 216.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 219.21 | 216.65 | 216.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 11:15:00 | 221.58 | 218.26 | 217.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 15:15:00 | 219.38 | 219.49 | 218.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:15:00 | 221.30 | 219.49 | 218.30 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 218.70 | 219.33 | 218.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 218.95 | 219.33 | 218.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 218.05 | 219.08 | 218.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 218.05 | 219.08 | 218.31 | SL hit (close<ema400) qty=1.00 sl=218.31 alert=retest1 |

### Cycle 191 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 216.55 | 217.86 | 217.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 215.30 | 217.35 | 217.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 213.75 | 213.53 | 214.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 11:15:00 | 213.75 | 213.53 | 214.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 213.75 | 213.53 | 214.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 214.55 | 213.53 | 214.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 213.60 | 213.68 | 214.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 212.50 | 213.79 | 214.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 13:15:00 | 201.88 | 205.49 | 207.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 204.80 | 204.58 | 206.66 | SL hit (close>ema200) qty=0.50 sl=204.58 alert=retest2 |

### Cycle 192 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 207.56 | 206.09 | 205.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 210.38 | 207.34 | 206.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 208.72 | 209.10 | 208.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 208.72 | 209.10 | 208.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 208.72 | 209.10 | 208.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 210.89 | 209.76 | 208.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 210.89 | 209.97 | 209.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 211.20 | 209.42 | 209.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 214.37 | 215.60 | 215.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 214.37 | 215.60 | 215.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 213.50 | 215.18 | 215.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 213.97 | 213.63 | 214.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 213.97 | 213.63 | 214.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 214.45 | 213.79 | 214.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 214.59 | 213.79 | 214.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 213.11 | 213.65 | 214.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 212.70 | 213.45 | 213.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 212.27 | 213.09 | 213.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 210.08 | 209.42 | 209.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 210.00 | 209.55 | 209.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 210.00 | 209.55 | 209.50 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 208.52 | 209.45 | 209.47 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 210.35 | 209.49 | 209.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 211.08 | 209.81 | 209.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 210.17 | 210.73 | 210.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 210.17 | 210.73 | 210.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 210.17 | 210.73 | 210.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 210.17 | 210.73 | 210.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 209.25 | 210.43 | 210.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 209.22 | 210.43 | 210.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 209.40 | 210.23 | 210.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 213.22 | 210.23 | 210.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 14:15:00 | 221.38 | 224.04 | 224.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 221.38 | 224.04 | 224.36 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 225.38 | 223.86 | 223.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 226.84 | 224.46 | 224.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 226.75 | 226.85 | 225.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:45:00 | 227.70 | 226.85 | 225.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 225.76 | 226.63 | 225.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 225.98 | 226.63 | 225.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 227.14 | 226.73 | 225.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:45:00 | 228.24 | 227.97 | 226.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 251.06 | 247.38 | 244.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 244.84 | 247.34 | 247.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 243.80 | 245.99 | 246.73 | Break + close below crossover candle low |

### Cycle 200 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 255.00 | 246.88 | 246.70 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 247.49 | 249.49 | 249.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 244.45 | 247.98 | 248.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 249.05 | 246.45 | 247.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 249.05 | 246.45 | 247.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 249.05 | 246.45 | 247.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 250.00 | 246.45 | 247.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 249.00 | 246.96 | 247.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 247.47 | 246.96 | 247.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 248.54 | 246.43 | 246.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 248.54 | 246.43 | 246.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 250.30 | 247.45 | 246.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 250.64 | 250.71 | 249.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 249.63 | 250.61 | 249.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 249.63 | 250.61 | 249.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 249.63 | 250.61 | 249.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 249.50 | 250.39 | 249.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 254.49 | 250.39 | 249.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 13:15:00 | 247.94 | 251.80 | 252.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 247.94 | 251.80 | 252.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 246.83 | 249.71 | 250.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 249.74 | 249.24 | 250.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 12:45:00 | 249.82 | 249.24 | 250.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 252.30 | 249.86 | 250.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 252.30 | 249.86 | 250.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 249.01 | 249.69 | 250.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 15:15:00 | 247.99 | 249.69 | 250.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 248.18 | 247.19 | 247.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 250.53 | 248.04 | 248.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 250.53 | 248.04 | 248.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 252.31 | 250.05 | 249.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 250.05 | 250.61 | 249.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:00:00 | 250.05 | 250.61 | 249.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 249.83 | 250.45 | 249.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 249.34 | 250.45 | 249.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 249.15 | 250.19 | 249.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 249.36 | 250.19 | 249.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 251.00 | 250.35 | 249.72 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 247.47 | 249.30 | 249.54 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 252.94 | 249.58 | 249.55 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 248.77 | 250.52 | 250.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 247.42 | 249.90 | 250.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 248.05 | 247.85 | 249.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 248.05 | 247.85 | 249.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 250.01 | 248.28 | 249.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 247.50 | 248.65 | 249.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:30:00 | 246.75 | 247.80 | 248.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:45:00 | 247.86 | 247.84 | 248.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 247.93 | 247.84 | 248.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 247.64 | 247.83 | 248.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 247.69 | 247.83 | 248.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 248.07 | 247.88 | 248.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 248.07 | 247.88 | 248.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 249.50 | 248.20 | 248.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 249.20 | 248.20 | 248.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 247.75 | 248.11 | 248.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:45:00 | 247.24 | 247.82 | 248.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 244.01 | 242.78 | 242.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 244.01 | 242.78 | 242.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 245.50 | 243.64 | 243.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 244.83 | 245.12 | 244.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:45:00 | 244.65 | 245.12 | 244.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 244.96 | 245.00 | 244.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 245.38 | 245.05 | 244.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 248.64 | 245.06 | 244.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 250.65 | 250.85 | 250.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 250.65 | 250.85 | 250.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 249.72 | 250.63 | 250.76 | Break + close below crossover candle low |

### Cycle 210 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 257.20 | 251.76 | 251.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 10:15:00 | 260.00 | 253.40 | 252.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 258.90 | 260.63 | 257.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 10:00:00 | 258.90 | 260.63 | 257.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 256.55 | 259.17 | 257.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 256.55 | 259.17 | 257.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 256.01 | 258.53 | 257.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 256.01 | 258.53 | 257.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 256.80 | 256.83 | 256.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 258.22 | 256.83 | 256.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 255.20 | 256.54 | 256.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 255.20 | 256.54 | 256.56 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 258.90 | 257.01 | 256.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 262.74 | 258.64 | 257.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 270.40 | 271.54 | 268.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:15:00 | 270.35 | 271.54 | 268.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 275.10 | 276.61 | 274.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 274.85 | 276.61 | 274.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 276.00 | 276.49 | 274.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 272.45 | 275.70 | 274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 271.20 | 274.80 | 274.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 271.20 | 274.80 | 274.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 270.40 | 273.34 | 273.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 269.10 | 271.13 | 272.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 13:15:00 | 271.20 | 271.01 | 271.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:00:00 | 271.20 | 271.01 | 271.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 274.75 | 271.76 | 272.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 274.40 | 271.76 | 272.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 273.90 | 272.19 | 272.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 273.75 | 272.19 | 272.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 272.15 | 272.21 | 272.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:30:00 | 270.60 | 271.40 | 271.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 271.00 | 265.68 | 265.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 271.00 | 265.68 | 265.40 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 263.50 | 266.37 | 266.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 260.20 | 263.09 | 264.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 256.15 | 255.81 | 259.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 262.30 | 255.81 | 259.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 263.40 | 257.33 | 259.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 263.40 | 257.33 | 259.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 263.80 | 258.62 | 260.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 264.90 | 258.62 | 260.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 261.25 | 260.02 | 260.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:30:00 | 261.80 | 260.02 | 260.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 264.40 | 260.89 | 260.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 14:15:00 | 264.70 | 262.23 | 261.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 299.95 | 300.26 | 294.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:45:00 | 299.95 | 300.26 | 294.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 288.10 | 297.94 | 295.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 289.00 | 297.94 | 295.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 289.70 | 296.29 | 295.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:30:00 | 287.75 | 296.29 | 295.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 286.50 | 294.33 | 294.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 283.25 | 292.11 | 293.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 292.85 | 289.89 | 291.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 292.85 | 289.89 | 291.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 292.85 | 289.89 | 291.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 292.30 | 289.89 | 291.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 293.05 | 290.52 | 291.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 293.05 | 290.52 | 291.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 295.55 | 291.53 | 292.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 295.55 | 291.53 | 292.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 292.60 | 291.74 | 292.21 | EMA400 retest candle locked (from downside) |

### Cycle 218 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 296.55 | 293.05 | 292.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 302.85 | 295.44 | 293.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 312.90 | 313.92 | 310.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 312.50 | 313.92 | 310.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 311.30 | 313.39 | 310.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 315.90 | 313.65 | 310.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 323.95 | 315.85 | 313.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 318.00 | 320.41 | 320.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 318.00 | 320.41 | 320.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 316.75 | 319.68 | 320.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 316.70 | 315.81 | 316.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 316.70 | 315.81 | 316.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 316.70 | 315.81 | 316.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:45:00 | 317.05 | 315.81 | 316.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 316.85 | 316.02 | 316.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 315.50 | 316.21 | 316.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 322.60 | 317.38 | 317.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 322.60 | 317.38 | 317.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 10:15:00 | 325.00 | 318.90 | 318.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 15:15:00 | 325.25 | 325.43 | 323.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:15:00 | 323.05 | 325.43 | 323.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 322.65 | 324.87 | 323.34 | EMA400 retest candle locked (from upside) |

### Cycle 221 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 320.10 | 322.53 | 322.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 318.85 | 321.80 | 322.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 323.70 | 322.18 | 322.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 323.70 | 322.18 | 322.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 323.70 | 322.18 | 322.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 324.00 | 322.18 | 322.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 323.00 | 322.34 | 322.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 324.85 | 322.34 | 322.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 323.60 | 322.56 | 322.55 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 13:15:00 | 322.30 | 322.51 | 322.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 321.25 | 322.26 | 322.41 | Break + close below crossover candle low |

### Cycle 224 — BUY (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 15:15:00 | 323.55 | 322.52 | 322.51 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 320.20 | 322.05 | 322.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 319.70 | 321.58 | 322.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 12:15:00 | 321.80 | 321.62 | 322.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 13:00:00 | 321.80 | 321.62 | 322.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 323.05 | 321.91 | 322.13 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 325.50 | 322.63 | 322.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 326.00 | 323.30 | 322.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 335.30 | 337.39 | 334.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 335.30 | 337.39 | 334.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 335.30 | 337.39 | 334.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 335.30 | 337.39 | 334.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 335.10 | 336.93 | 334.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 334.45 | 336.93 | 334.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 333.85 | 336.32 | 334.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:15:00 | 329.60 | 336.32 | 334.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 326.25 | 334.30 | 334.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 326.25 | 334.30 | 334.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 322.35 | 331.91 | 333.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 10:15:00 | 320.00 | 326.17 | 329.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 314.85 | 314.15 | 318.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 15:15:00 | 317.00 | 315.57 | 317.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 317.00 | 315.57 | 317.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 317.80 | 315.57 | 317.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 316.50 | 315.76 | 317.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 314.15 | 315.63 | 316.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 298.44 | 310.56 | 314.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 300.30 | 298.08 | 303.84 | SL hit (close>ema200) qty=0.50 sl=298.08 alert=retest2 |

### Cycle 228 — BUY (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 14:15:00 | 305.40 | 304.24 | 304.17 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 298.95 | 303.33 | 303.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 10:15:00 | 298.05 | 302.27 | 303.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 284.55 | 282.82 | 287.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 284.30 | 282.82 | 287.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 283.75 | 283.20 | 287.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 277.25 | 284.52 | 285.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 279.60 | 282.82 | 284.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 263.39 | 269.29 | 274.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 265.62 | 269.29 | 274.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 261.95 | 261.69 | 267.84 | SL hit (close>ema200) qty=0.50 sl=261.69 alert=retest2 |

### Cycle 230 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 273.00 | 268.02 | 267.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 285.60 | 272.19 | 269.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 281.25 | 282.51 | 277.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 281.25 | 282.51 | 277.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 281.25 | 282.51 | 277.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:00:00 | 284.25 | 282.86 | 278.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 11:45:00 | 284.20 | 286.69 | 283.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:00:00 | 284.75 | 286.30 | 283.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 14:45:00 | 288.45 | 286.46 | 284.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 271.20 | 287.44 | 286.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 271.20 | 287.44 | 286.95 | SL hit (close<static) qty=1.00 sl=277.10 alert=retest2 |

### Cycle 231 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 271.20 | 284.19 | 285.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 267.95 | 274.08 | 278.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 271.35 | 270.60 | 274.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:15:00 | 268.85 | 270.60 | 274.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 275.90 | 267.23 | 270.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 275.30 | 267.23 | 270.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 274.00 | 268.59 | 270.56 | EMA400 retest candle locked (from downside) |

### Cycle 232 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 278.70 | 271.83 | 271.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 13:15:00 | 284.70 | 274.40 | 272.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 277.25 | 279.48 | 276.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 13:15:00 | 277.25 | 279.48 | 276.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 277.25 | 279.48 | 276.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:00:00 | 277.25 | 279.48 | 276.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 277.80 | 279.15 | 277.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 278.20 | 279.15 | 277.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 277.60 | 278.84 | 277.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 279.85 | 278.84 | 277.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 278.70 | 281.48 | 279.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 278.95 | 280.13 | 279.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:15:00 | 278.35 | 280.60 | 280.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 281.05 | 280.83 | 280.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 281.05 | 280.83 | 280.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 280.00 | 280.66 | 280.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 280.45 | 280.66 | 280.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 278.95 | 280.32 | 280.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 09:15:00 | 278.95 | 280.32 | 280.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 10:15:00 | 276.50 | 279.55 | 280.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 10:15:00 | 278.20 | 277.77 | 278.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 10:15:00 | 278.20 | 277.77 | 278.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 278.20 | 277.77 | 278.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:30:00 | 279.20 | 277.77 | 278.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 276.80 | 277.62 | 278.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:15:00 | 275.30 | 277.62 | 278.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:00:00 | 275.85 | 277.26 | 278.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:30:00 | 275.50 | 276.89 | 277.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 284.55 | 278.33 | 278.35 | SL hit (close>static) qty=1.00 sl=278.70 alert=retest2 |

### Cycle 234 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 287.55 | 280.18 | 279.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 14:15:00 | 290.50 | 284.65 | 281.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 297.05 | 297.24 | 293.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 14:00:00 | 297.05 | 297.24 | 293.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 294.85 | 296.78 | 294.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 293.30 | 296.78 | 294.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 295.20 | 296.46 | 294.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 295.50 | 296.46 | 294.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 295.00 | 296.17 | 294.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 294.65 | 296.17 | 294.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 295.30 | 296.00 | 294.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 298.15 | 295.61 | 294.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 293.65 | 296.48 | 296.34 | SL hit (close<static) qty=1.00 sl=294.20 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 292.20 | 295.63 | 295.96 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 297.80 | 296.10 | 295.92 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 291.85 | 295.10 | 295.51 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 301.50 | 295.86 | 295.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 304.90 | 297.67 | 296.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 13:15:00 | 298.00 | 298.05 | 296.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 13:30:00 | 297.95 | 298.05 | 296.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 297.30 | 297.90 | 297.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:45:00 | 297.15 | 297.90 | 297.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 298.95 | 298.11 | 297.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 300.65 | 297.92 | 297.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:00:00 | 301.00 | 298.54 | 297.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-24 09:30:00 | 160.13 | 2024-04-24 10:15:00 | 159.17 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-04-25 11:30:00 | 159.17 | 2024-04-26 09:15:00 | 160.54 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-04-25 14:45:00 | 159.42 | 2024-04-26 09:15:00 | 160.54 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-05-09 15:15:00 | 155.83 | 2024-05-14 09:15:00 | 160.38 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-05-13 09:45:00 | 156.67 | 2024-05-14 09:15:00 | 160.38 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-05-24 09:15:00 | 165.71 | 2024-05-24 11:15:00 | 164.04 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-05-31 10:00:00 | 161.54 | 2024-05-31 14:15:00 | 165.54 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-05-31 10:45:00 | 162.21 | 2024-05-31 14:15:00 | 165.54 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-06-25 10:30:00 | 173.58 | 2024-06-28 14:15:00 | 172.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-07-15 12:30:00 | 165.84 | 2024-07-26 09:15:00 | 182.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 09:15:00 | 187.08 | 2024-08-23 09:15:00 | 185.38 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-08-30 11:45:00 | 186.73 | 2024-08-30 12:15:00 | 187.12 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-09-05 14:00:00 | 184.42 | 2024-09-10 13:15:00 | 186.17 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-09-11 13:15:00 | 184.48 | 2024-09-11 14:15:00 | 182.48 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-13 10:15:00 | 181.65 | 2024-09-19 10:15:00 | 172.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 10:15:00 | 181.65 | 2024-09-19 13:15:00 | 176.29 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2024-10-07 10:15:00 | 167.08 | 2024-10-14 12:15:00 | 167.24 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-10-07 11:30:00 | 167.80 | 2024-10-14 12:15:00 | 167.24 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-10-15 14:30:00 | 169.42 | 2024-10-18 15:15:00 | 186.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-16 10:00:00 | 168.36 | 2024-10-18 15:15:00 | 185.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-16 14:45:00 | 168.40 | 2024-10-18 15:15:00 | 185.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-17 11:30:00 | 168.65 | 2024-10-18 15:15:00 | 185.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-17 13:45:00 | 171.48 | 2024-10-21 13:15:00 | 188.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-11 10:30:00 | 191.22 | 2024-11-11 14:15:00 | 189.19 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest1 | 2024-11-14 14:15:00 | 176.89 | 2024-11-18 13:15:00 | 179.45 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-03 09:15:00 | 200.16 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-12-03 12:00:00 | 199.22 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-12-03 13:15:00 | 199.06 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2024-12-04 09:15:00 | 199.85 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-12-05 10:30:00 | 201.66 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-12-05 11:00:00 | 203.08 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-12-06 10:15:00 | 203.75 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-12-06 14:15:00 | 202.49 | 2024-12-11 12:15:00 | 199.98 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest1 | 2024-12-13 09:15:00 | 195.94 | 2024-12-13 14:15:00 | 198.79 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-12-13 10:15:00 | 194.30 | 2024-12-16 12:15:00 | 199.23 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-12-17 12:00:00 | 194.58 | 2024-12-20 14:15:00 | 184.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 12:30:00 | 194.58 | 2024-12-20 14:15:00 | 184.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 15:15:00 | 194.00 | 2024-12-20 14:15:00 | 184.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 12:00:00 | 194.58 | 2024-12-27 09:15:00 | 180.44 | STOP_HIT | 0.50 | 7.27% |
| SELL | retest2 | 2024-12-17 12:30:00 | 194.58 | 2024-12-27 09:15:00 | 180.44 | STOP_HIT | 0.50 | 7.27% |
| SELL | retest2 | 2024-12-17 15:15:00 | 194.00 | 2024-12-27 09:15:00 | 180.44 | STOP_HIT | 0.50 | 6.99% |
| SELL | retest2 | 2024-12-20 13:30:00 | 188.32 | 2024-12-27 14:15:00 | 186.03 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2025-01-08 10:30:00 | 178.17 | 2025-01-14 12:15:00 | 179.78 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-01-08 14:15:00 | 178.64 | 2025-01-14 12:15:00 | 179.78 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-01-08 14:45:00 | 177.76 | 2025-01-14 12:15:00 | 179.78 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-01-16 09:15:00 | 177.85 | 2025-01-27 09:15:00 | 185.57 | STOP_HIT | 1.00 | 4.34% |
| SELL | retest2 | 2025-02-14 11:00:00 | 185.10 | 2025-02-24 09:15:00 | 175.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 12:00:00 | 184.66 | 2025-02-24 09:15:00 | 175.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 10:00:00 | 184.42 | 2025-02-24 09:15:00 | 175.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 11:00:00 | 184.36 | 2025-02-24 09:15:00 | 175.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 11:00:00 | 185.10 | 2025-02-25 10:15:00 | 176.04 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-02-14 12:00:00 | 184.66 | 2025-02-25 10:15:00 | 176.04 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2025-02-17 10:00:00 | 184.42 | 2025-02-25 10:15:00 | 176.04 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-02-17 11:00:00 | 184.36 | 2025-02-25 10:15:00 | 176.04 | STOP_HIT | 0.50 | 4.51% |
| SELL | retest2 | 2025-02-19 13:00:00 | 180.00 | 2025-02-28 09:15:00 | 171.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 11:45:00 | 180.15 | 2025-02-28 09:15:00 | 171.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 10:00:00 | 179.43 | 2025-02-28 09:15:00 | 170.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 13:00:00 | 180.00 | 2025-03-03 09:15:00 | 162.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-20 11:45:00 | 180.15 | 2025-03-03 09:15:00 | 162.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-21 10:00:00 | 179.43 | 2025-03-03 09:15:00 | 161.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-26 13:15:00 | 175.07 | 2025-03-27 15:15:00 | 173.21 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-03-27 10:00:00 | 175.09 | 2025-03-27 15:15:00 | 173.21 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-03-27 12:00:00 | 175.08 | 2025-03-27 15:15:00 | 173.21 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-03-28 09:15:00 | 176.29 | 2025-03-28 14:15:00 | 174.18 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-05-06 11:00:00 | 175.83 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-05-06 14:45:00 | 174.68 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-05-07 09:45:00 | 176.03 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-05-07 13:15:00 | 176.01 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-05-08 11:30:00 | 175.96 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-05-08 13:45:00 | 176.53 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-05-12 10:00:00 | 176.93 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-05-12 11:45:00 | 176.97 | 2025-05-12 13:15:00 | 175.83 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-05-21 14:15:00 | 189.37 | 2025-05-22 09:15:00 | 187.03 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-30 10:15:00 | 182.67 | 2025-06-02 09:15:00 | 192.42 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-05-30 11:15:00 | 182.25 | 2025-06-02 09:15:00 | 192.42 | STOP_HIT | 1.00 | -5.58% |
| SELL | retest2 | 2025-05-30 12:15:00 | 182.57 | 2025-06-02 09:15:00 | 192.42 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest2 | 2025-06-11 10:45:00 | 201.30 | 2025-06-19 11:15:00 | 203.62 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest1 | 2025-07-14 09:15:00 | 222.75 | 2025-07-15 09:15:00 | 226.67 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-07-21 11:45:00 | 225.50 | 2025-07-22 11:15:00 | 226.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-25 09:15:00 | 222.92 | 2025-07-31 12:15:00 | 218.38 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2025-08-22 11:30:00 | 216.25 | 2025-08-25 09:15:00 | 219.21 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-22 14:30:00 | 216.54 | 2025-08-25 09:15:00 | 219.21 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest1 | 2025-08-26 09:15:00 | 221.30 | 2025-08-26 10:15:00 | 218.05 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-29 15:15:00 | 212.50 | 2025-09-04 13:15:00 | 201.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 15:15:00 | 212.50 | 2025-09-05 09:15:00 | 204.80 | STOP_HIT | 0.50 | 3.62% |
| BUY | retest2 | 2025-09-12 15:15:00 | 210.89 | 2025-09-22 12:15:00 | 214.37 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-09-15 11:15:00 | 210.89 | 2025-09-22 12:15:00 | 214.37 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-09-16 14:15:00 | 211.20 | 2025-09-22 12:15:00 | 214.37 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-09-24 14:45:00 | 212.70 | 2025-09-30 15:15:00 | 210.00 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2025-09-25 10:30:00 | 212.27 | 2025-09-30 15:15:00 | 210.00 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-09-30 09:15:00 | 210.08 | 2025-09-30 15:15:00 | 210.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-10-06 09:15:00 | 213.22 | 2025-10-14 14:15:00 | 221.38 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2025-10-20 09:45:00 | 228.24 | 2025-10-28 09:15:00 | 251.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-10 09:15:00 | 247.47 | 2025-11-12 12:15:00 | 248.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-11-17 09:15:00 | 254.49 | 2025-11-20 13:15:00 | 247.94 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-11-21 15:15:00 | 247.99 | 2025-11-26 09:15:00 | 250.53 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-11-25 14:30:00 | 248.18 | 2025-11-26 09:15:00 | 250.53 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-04 13:15:00 | 247.50 | 2025-12-12 11:15:00 | 244.01 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2025-12-05 10:30:00 | 246.75 | 2025-12-12 11:15:00 | 244.01 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-12-05 11:45:00 | 247.86 | 2025-12-12 11:15:00 | 244.01 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-12-05 12:15:00 | 247.93 | 2025-12-12 11:15:00 | 244.01 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2025-12-08 10:45:00 | 247.24 | 2025-12-12 11:15:00 | 244.01 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-12-16 14:45:00 | 245.38 | 2025-12-24 11:15:00 | 250.65 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-12-17 09:15:00 | 248.64 | 2025-12-24 11:15:00 | 250.65 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-12-30 11:15:00 | 258.22 | 2025-12-30 13:15:00 | 255.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-09 13:30:00 | 270.60 | 2026-01-14 15:15:00 | 271.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-02-06 11:15:00 | 315.90 | 2026-02-12 10:15:00 | 318.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-02-09 09:15:00 | 323.95 | 2026-02-12 10:15:00 | 318.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-16 15:15:00 | 315.50 | 2026-02-17 09:15:00 | 322.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-03-06 13:15:00 | 314.15 | 2026-03-09 09:15:00 | 298.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 314.15 | 2026-03-10 10:15:00 | 300.30 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2026-03-19 09:15:00 | 277.25 | 2026-03-20 14:15:00 | 263.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 10:30:00 | 279.60 | 2026-03-20 14:15:00 | 265.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 277.25 | 2026-03-23 13:15:00 | 261.95 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2026-03-19 10:30:00 | 279.60 | 2026-03-23 13:15:00 | 261.95 | STOP_HIT | 0.50 | 6.31% |
| BUY | retest2 | 2026-03-27 11:00:00 | 284.25 | 2026-04-02 09:15:00 | 271.20 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2026-03-30 11:45:00 | 284.20 | 2026-04-02 09:15:00 | 271.20 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest2 | 2026-03-30 13:00:00 | 284.75 | 2026-04-02 09:15:00 | 271.20 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2026-03-30 14:45:00 | 288.45 | 2026-04-02 09:15:00 | 271.20 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest2 | 2026-04-10 09:15:00 | 279.85 | 2026-04-17 09:15:00 | 278.95 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2026-04-13 10:15:00 | 278.70 | 2026-04-17 09:15:00 | 278.95 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2026-04-13 11:45:00 | 278.95 | 2026-04-17 09:15:00 | 278.95 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-04-16 13:15:00 | 278.35 | 2026-04-17 09:15:00 | 278.95 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2026-04-20 13:15:00 | 275.30 | 2026-04-21 10:15:00 | 284.55 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-04-20 14:00:00 | 275.85 | 2026-04-21 10:15:00 | 284.55 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-04-20 14:30:00 | 275.50 | 2026-04-21 10:15:00 | 284.55 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2026-04-27 09:15:00 | 298.15 | 2026-04-28 13:15:00 | 293.65 | STOP_HIT | 1.00 | -1.51% |
