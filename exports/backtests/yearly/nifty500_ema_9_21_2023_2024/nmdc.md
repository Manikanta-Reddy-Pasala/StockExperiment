# NMDC Ltd. (NMDC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 88.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 206 |
| ALERT1 | 154 |
| ALERT2 | 152 |
| ALERT2_SKIP | 87 |
| ALERT3 | 332 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 102 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 109 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 84
- **Target hits / Stop hits / Partials:** 0 / 109 / 8
- **Avg / median % per leg:** -0.44% / -1.31%
- **Sum % (uncompounded):** -51.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 16 | 34.8% | 0 | 46 | 0 | -0.75% | -34.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.42% | -1.4% |
| BUY @ 3rd Alert (retest2) | 45 | 16 | 35.6% | 0 | 45 | 0 | -0.74% | -33.3% |
| SELL (all) | 71 | 17 | 23.9% | 0 | 63 | 8 | -0.24% | -16.9% |
| SELL @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 0 | 6 | 3 | 2.58% | 23.2% |
| SELL @ 3rd Alert (retest2) | 62 | 11 | 17.7% | 0 | 57 | 5 | -0.65% | -40.1% |
| retest1 (combined) | 10 | 6 | 60.0% | 0 | 7 | 3 | 2.18% | 21.8% |
| retest2 (combined) | 107 | 27 | 25.2% | 0 | 102 | 5 | -0.69% | -73.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 35.52 | 35.29 | 35.27 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 35.12 | 35.26 | 35.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 14:15:00 | 35.05 | 35.20 | 35.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 35.07 | 34.92 | 35.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 35.07 | 34.92 | 35.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 35.07 | 34.92 | 35.03 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 12:15:00 | 35.32 | 35.12 | 35.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 13:15:00 | 35.48 | 35.19 | 35.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 15:15:00 | 35.62 | 35.63 | 35.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 35.77 | 35.66 | 35.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 35.77 | 35.66 | 35.49 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 36.20 | 36.39 | 36.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 35.85 | 36.28 | 36.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 35.95 | 35.94 | 36.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 10:15:00 | 36.07 | 35.96 | 36.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 36.07 | 35.96 | 36.09 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 36.33 | 36.13 | 36.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 11:15:00 | 36.50 | 36.20 | 36.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 15:15:00 | 36.28 | 36.29 | 36.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 15:15:00 | 36.28 | 36.29 | 36.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 36.28 | 36.29 | 36.22 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 36.07 | 36.23 | 36.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 13:15:00 | 35.98 | 36.15 | 36.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 09:15:00 | 35.85 | 35.63 | 35.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 35.85 | 35.63 | 35.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 35.85 | 35.63 | 35.78 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 35.67 | 35.04 | 34.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 13:15:00 | 35.83 | 35.46 | 35.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 15:15:00 | 35.70 | 35.71 | 35.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 13:15:00 | 35.73 | 35.75 | 35.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 13:15:00 | 35.73 | 35.75 | 35.63 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 35.62 | 35.66 | 35.67 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 35.88 | 35.71 | 35.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 10:15:00 | 36.17 | 35.80 | 35.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 12:15:00 | 35.93 | 35.93 | 35.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 37.53 | 37.76 | 37.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 37.53 | 37.76 | 37.51 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 14:15:00 | 37.13 | 37.37 | 37.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 10:15:00 | 36.93 | 37.07 | 37.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 11:15:00 | 37.12 | 37.08 | 37.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 11:15:00 | 37.12 | 37.08 | 37.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 37.12 | 37.08 | 37.16 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 37.57 | 37.13 | 37.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 10:15:00 | 37.60 | 37.41 | 37.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 14:15:00 | 37.47 | 37.51 | 37.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 14:15:00 | 37.43 | 37.53 | 37.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 37.43 | 37.53 | 37.46 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 10:15:00 | 37.25 | 37.40 | 37.41 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 15:15:00 | 37.52 | 37.43 | 37.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 37.92 | 37.53 | 37.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 38.77 | 38.89 | 38.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 38.77 | 38.89 | 38.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 38.77 | 38.89 | 38.55 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 37.58 | 38.34 | 38.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 10:15:00 | 37.43 | 37.77 | 37.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 37.37 | 37.26 | 37.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 37.37 | 37.26 | 37.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 37.37 | 37.26 | 37.46 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 38.00 | 37.53 | 37.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 09:15:00 | 38.23 | 37.85 | 37.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 38.98 | 39.00 | 38.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 38.58 | 38.93 | 38.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 38.58 | 38.93 | 38.73 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 12:15:00 | 38.18 | 38.65 | 38.69 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 39.05 | 38.70 | 38.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 12:15:00 | 39.25 | 38.86 | 38.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 10:15:00 | 40.62 | 40.73 | 40.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 11:15:00 | 40.23 | 40.63 | 40.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 40.23 | 40.63 | 40.30 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 39.97 | 40.21 | 40.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 13:15:00 | 39.90 | 40.15 | 40.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 40.42 | 40.11 | 40.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 40.42 | 40.11 | 40.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 40.42 | 40.11 | 40.16 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 40.73 | 40.23 | 40.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 13:15:00 | 40.80 | 40.48 | 40.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 41.15 | 41.16 | 40.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 40.58 | 41.06 | 40.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 40.58 | 41.06 | 40.95 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 14:15:00 | 40.88 | 40.89 | 40.89 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 15:15:00 | 41.00 | 40.91 | 40.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 41.82 | 41.09 | 40.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 13:15:00 | 46.78 | 46.83 | 45.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 47.32 | 47.50 | 47.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 47.32 | 47.50 | 47.22 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 46.25 | 47.39 | 47.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 45.93 | 47.09 | 47.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 46.78 | 46.29 | 46.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 46.78 | 46.29 | 46.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 46.78 | 46.29 | 46.70 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 49.02 | 47.31 | 47.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 10:15:00 | 49.73 | 47.79 | 47.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 15:15:00 | 49.77 | 49.82 | 49.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 49.48 | 49.75 | 49.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 49.48 | 49.75 | 49.16 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 48.92 | 49.29 | 49.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 11:15:00 | 48.12 | 49.02 | 49.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 48.00 | 47.97 | 48.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 14:15:00 | 47.80 | 47.65 | 47.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 47.80 | 47.65 | 47.95 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 48.17 | 47.79 | 47.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 12:15:00 | 49.10 | 48.05 | 47.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 48.57 | 49.19 | 48.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 48.57 | 49.19 | 48.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 48.57 | 49.19 | 48.89 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 11:15:00 | 48.58 | 48.81 | 48.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 47.60 | 48.51 | 48.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 48.03 | 47.85 | 48.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 48.03 | 47.85 | 48.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 48.03 | 47.85 | 48.17 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 48.87 | 48.33 | 48.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 14:15:00 | 50.05 | 48.68 | 48.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 53.22 | 53.34 | 52.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 53.72 | 54.66 | 54.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 53.72 | 54.66 | 54.27 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 53.65 | 54.02 | 54.07 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 13:15:00 | 54.57 | 54.10 | 54.08 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 09:15:00 | 53.50 | 54.05 | 54.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 10:15:00 | 53.08 | 53.86 | 53.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 51.33 | 51.30 | 52.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 50.93 | 50.65 | 51.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 50.93 | 50.65 | 51.03 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 51.43 | 51.22 | 51.20 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 10:15:00 | 51.17 | 51.19 | 51.19 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 51.47 | 51.24 | 51.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 14:15:00 | 52.00 | 51.47 | 51.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 10:15:00 | 51.38 | 51.54 | 51.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 10:15:00 | 51.38 | 51.54 | 51.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 51.38 | 51.54 | 51.41 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 51.32 | 51.41 | 51.42 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 51.93 | 51.47 | 51.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 11:15:00 | 52.10 | 51.66 | 51.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 54.05 | 54.18 | 53.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 10:15:00 | 53.98 | 54.13 | 53.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 53.98 | 54.13 | 53.94 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 11:15:00 | 56.23 | 56.74 | 56.74 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 11:15:00 | 56.65 | 56.53 | 56.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 14:15:00 | 56.92 | 56.62 | 56.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 09:15:00 | 56.32 | 56.61 | 56.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 56.32 | 56.61 | 56.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 56.32 | 56.61 | 56.57 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 10:15:00 | 56.30 | 56.55 | 56.55 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 11:15:00 | 57.00 | 56.64 | 56.59 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 56.37 | 56.59 | 56.59 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 56.82 | 56.64 | 56.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 14:15:00 | 57.18 | 56.76 | 56.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 57.70 | 58.13 | 57.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 14:15:00 | 57.70 | 58.13 | 57.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 57.70 | 58.13 | 57.70 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 61.05 | 61.32 | 61.32 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 61.65 | 61.38 | 61.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 10:15:00 | 61.88 | 61.56 | 61.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 61.48 | 61.57 | 61.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 12:15:00 | 61.48 | 61.57 | 61.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 61.48 | 61.57 | 61.49 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 61.02 | 61.42 | 61.44 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 09:15:00 | 62.08 | 61.49 | 61.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 10:15:00 | 62.68 | 61.73 | 61.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 65.10 | 65.13 | 64.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 64.15 | 64.96 | 64.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 64.15 | 64.96 | 64.63 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 62.45 | 64.48 | 64.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 61.13 | 63.81 | 64.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 10:15:00 | 63.15 | 63.12 | 63.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 12:15:00 | 63.88 | 63.28 | 63.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 63.88 | 63.28 | 63.75 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 64.93 | 64.04 | 63.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 66.77 | 65.17 | 64.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 67.02 | 67.04 | 66.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 66.70 | 66.97 | 66.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 66.70 | 66.97 | 66.26 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 13:15:00 | 72.77 | 73.34 | 73.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 71.58 | 72.99 | 73.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 12:15:00 | 72.87 | 72.74 | 72.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 13:15:00 | 72.72 | 72.74 | 72.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 72.72 | 72.74 | 72.97 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 70.27 | 68.99 | 68.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 10:15:00 | 72.30 | 70.03 | 69.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 70.00 | 70.75 | 70.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 70.00 | 70.75 | 70.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 70.00 | 70.75 | 70.22 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 68.08 | 69.62 | 69.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 67.52 | 69.20 | 69.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 69.63 | 69.06 | 69.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 69.63 | 69.06 | 69.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 69.63 | 69.06 | 69.45 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 70.53 | 69.76 | 69.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 15:15:00 | 70.83 | 69.98 | 69.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 73.17 | 73.62 | 72.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 12:15:00 | 73.10 | 73.35 | 73.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 73.10 | 73.35 | 73.04 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 79.12 | 80.30 | 80.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 77.43 | 79.30 | 79.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 76.62 | 76.27 | 77.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 12:15:00 | 77.18 | 76.55 | 77.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 77.18 | 76.55 | 77.18 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 83.75 | 78.67 | 78.02 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 13:15:00 | 80.17 | 81.37 | 81.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 09:15:00 | 79.12 | 80.65 | 81.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 79.00 | 78.85 | 79.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 79.68 | 79.16 | 79.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 79.68 | 79.16 | 79.57 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 77.07 | 76.05 | 75.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 77.23 | 76.29 | 76.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 78.82 | 80.09 | 79.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 78.82 | 80.09 | 79.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 78.82 | 80.09 | 79.33 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 77.87 | 79.50 | 79.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 11:15:00 | 77.18 | 78.69 | 79.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 11:15:00 | 66.27 | 65.95 | 67.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 13:15:00 | 67.33 | 66.40 | 67.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 67.33 | 66.40 | 67.41 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 11:15:00 | 68.15 | 67.56 | 67.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 69.15 | 68.29 | 67.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 11:15:00 | 68.33 | 68.43 | 68.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 09:15:00 | 67.92 | 68.59 | 68.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 67.92 | 68.59 | 68.30 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 15:15:00 | 68.00 | 68.19 | 68.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 09:15:00 | 67.72 | 67.92 | 68.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 67.42 | 67.18 | 67.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 67.42 | 67.18 | 67.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 67.42 | 67.18 | 67.53 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 70.07 | 67.91 | 67.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 70.53 | 69.02 | 68.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 71.95 | 72.29 | 71.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 10:15:00 | 71.67 | 72.17 | 71.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 71.67 | 72.17 | 71.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 80.72 | 79.43 | 77.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 14:15:00 | 78.12 | 79.94 | 80.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 78.12 | 79.94 | 80.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 77.33 | 79.00 | 79.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 78.78 | 78.76 | 79.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 14:00:00 | 78.78 | 78.76 | 79.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 79.22 | 78.81 | 79.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 09:30:00 | 78.60 | 79.05 | 79.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 11:00:00 | 78.80 | 79.00 | 79.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 12:45:00 | 78.82 | 78.94 | 79.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 14:15:00 | 78.67 | 78.92 | 79.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 79.78 | 78.87 | 78.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 80.32 | 79.16 | 79.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 80.32 | 79.16 | 79.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 11:15:00 | 81.10 | 79.55 | 79.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 85.02 | 85.22 | 83.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 10:00:00 | 85.02 | 85.22 | 83.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 84.80 | 85.25 | 84.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 84.80 | 85.25 | 84.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 84.83 | 85.16 | 84.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 84.95 | 85.16 | 84.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 85.95 | 85.32 | 84.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:15:00 | 86.68 | 85.50 | 85.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 13:15:00 | 86.35 | 85.62 | 85.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 13:15:00 | 86.90 | 88.02 | 87.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 14:15:00 | 86.97 | 87.74 | 87.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 14:15:00 | 86.97 | 87.74 | 87.81 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 09:15:00 | 88.78 | 87.85 | 87.84 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 87.05 | 87.91 | 88.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 85.88 | 87.50 | 87.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 15:15:00 | 85.32 | 85.27 | 86.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 09:15:00 | 83.87 | 85.27 | 86.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 85.08 | 84.59 | 85.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 85.50 | 84.59 | 85.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 85.37 | 84.75 | 85.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-13 15:15:00 | 85.37 | 84.75 | 85.26 | SL hit (close>ema400) qty=1.00 sl=85.26 alert=retest1 |

### Cycle 65 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 87.68 | 85.66 | 85.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 88.32 | 86.51 | 86.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 11:15:00 | 88.85 | 88.94 | 88.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:30:00 | 88.78 | 88.94 | 88.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 87.33 | 88.57 | 88.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 87.53 | 88.57 | 88.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 88.33 | 88.52 | 88.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 89.38 | 88.49 | 88.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 14:15:00 | 91.02 | 91.83 | 91.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 91.02 | 91.83 | 91.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 09:15:00 | 89.67 | 91.32 | 91.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 89.93 | 89.39 | 90.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:30:00 | 89.68 | 89.39 | 90.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 90.32 | 89.68 | 90.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 90.32 | 89.68 | 90.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 89.27 | 89.60 | 90.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 88.45 | 89.20 | 89.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:15:00 | 84.03 | 85.77 | 86.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 84.55 | 84.19 | 85.16 | SL hit (close>ema200) qty=0.50 sl=84.19 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 86.67 | 85.71 | 85.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 89.28 | 86.43 | 85.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 86.85 | 88.24 | 87.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 86.85 | 88.24 | 87.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 86.85 | 88.24 | 87.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 86.13 | 88.24 | 87.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 80.87 | 86.76 | 86.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 75.92 | 84.60 | 85.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 81.10 | 80.31 | 82.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 81.10 | 80.31 | 82.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 81.83 | 80.80 | 82.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 81.90 | 80.80 | 82.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 84.55 | 81.65 | 82.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 85.08 | 81.65 | 82.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 85.48 | 82.42 | 82.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 85.83 | 82.42 | 82.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 84.77 | 82.89 | 82.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 85.95 | 84.08 | 83.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 85.47 | 85.51 | 84.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:45:00 | 85.78 | 85.51 | 84.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 84.90 | 85.34 | 84.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 85.82 | 85.26 | 84.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 86.35 | 85.64 | 85.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 87.78 | 88.19 | 88.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 87.78 | 88.19 | 88.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 87.33 | 87.97 | 88.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 89.33 | 88.15 | 88.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 89.33 | 88.15 | 88.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 89.33 | 88.15 | 88.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 89.33 | 88.15 | 88.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 90.58 | 88.64 | 88.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 91.55 | 89.22 | 88.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 89.82 | 90.44 | 89.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 12:15:00 | 89.82 | 90.44 | 89.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 89.82 | 90.44 | 89.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 89.82 | 90.44 | 89.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 90.33 | 90.42 | 89.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:15:00 | 89.83 | 90.42 | 89.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 89.90 | 90.32 | 89.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 89.75 | 90.32 | 89.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 89.90 | 90.23 | 89.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 88.68 | 90.23 | 89.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 88.62 | 89.91 | 89.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 87.70 | 89.91 | 89.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 88.73 | 89.54 | 89.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 12:15:00 | 88.12 | 89.26 | 89.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 82.67 | 82.38 | 83.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 09:30:00 | 82.97 | 82.38 | 83.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 83.19 | 82.53 | 83.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 83.20 | 82.53 | 83.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 83.66 | 82.75 | 83.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:45:00 | 83.68 | 82.75 | 83.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 83.53 | 82.91 | 83.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 83.21 | 82.91 | 83.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:45:00 | 83.24 | 82.92 | 83.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 14:15:00 | 83.93 | 83.17 | 83.26 | SL hit (close>static) qty=1.00 sl=83.90 alert=retest2 |

### Cycle 73 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 84.03 | 83.34 | 83.33 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 82.62 | 83.20 | 83.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 10:15:00 | 82.13 | 82.99 | 83.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 82.77 | 82.08 | 82.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 82.77 | 82.08 | 82.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 82.77 | 82.08 | 82.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:30:00 | 82.81 | 82.08 | 82.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 82.80 | 82.22 | 82.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:45:00 | 82.95 | 82.22 | 82.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 83.84 | 82.82 | 82.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 84.16 | 83.23 | 82.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 83.33 | 83.65 | 83.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 83.33 | 83.65 | 83.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 83.33 | 83.65 | 83.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 83.15 | 83.65 | 83.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 83.89 | 83.70 | 83.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 83.42 | 83.70 | 83.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 83.82 | 83.88 | 83.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 83.67 | 83.88 | 83.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 83.60 | 83.86 | 83.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 83.51 | 83.86 | 83.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 84.06 | 83.90 | 83.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:30:00 | 84.48 | 83.79 | 83.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 81.60 | 83.41 | 83.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 81.60 | 83.41 | 83.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 81.28 | 82.99 | 83.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 10:15:00 | 81.93 | 81.88 | 82.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 11:00:00 | 81.93 | 81.88 | 82.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 82.80 | 82.14 | 82.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 82.80 | 82.14 | 82.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 82.68 | 82.25 | 82.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:15:00 | 82.36 | 82.25 | 82.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 15:15:00 | 82.48 | 82.35 | 82.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 15:15:00 | 82.48 | 82.35 | 82.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 82.81 | 82.44 | 82.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 82.57 | 82.79 | 82.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 82.57 | 82.79 | 82.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 82.57 | 82.79 | 82.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 82.57 | 82.79 | 82.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 82.21 | 82.67 | 82.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 82.17 | 82.67 | 82.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 82.34 | 82.61 | 82.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 81.59 | 82.61 | 82.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 81.82 | 82.45 | 82.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 77.80 | 80.51 | 81.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 77.68 | 77.65 | 79.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 77.68 | 77.65 | 79.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 77.28 | 77.78 | 78.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 76.99 | 77.78 | 78.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 76.95 | 77.72 | 78.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:45:00 | 77.11 | 77.68 | 78.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 74.76 | 77.68 | 78.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 78.20 | 77.69 | 78.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 78.15 | 77.69 | 78.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 78.17 | 77.79 | 78.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 78.79 | 77.79 | 78.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 78.38 | 77.91 | 78.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 78.65 | 77.91 | 78.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 78.35 | 77.99 | 78.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 78.62 | 77.99 | 78.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 78.18 | 78.03 | 78.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:30:00 | 77.69 | 78.00 | 78.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:45:00 | 77.83 | 77.94 | 78.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 11:30:00 | 77.70 | 77.64 | 77.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 78.94 | 77.43 | 77.63 | SL hit (close>static) qty=1.00 sl=78.73 alert=retest2 |

### Cycle 79 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 79.13 | 78.02 | 77.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 79.41 | 78.48 | 78.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 80.98 | 81.07 | 80.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 80.98 | 81.07 | 80.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 80.57 | 80.83 | 80.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:45:00 | 80.48 | 80.83 | 80.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 80.60 | 80.78 | 80.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:30:00 | 80.57 | 80.78 | 80.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 80.49 | 80.73 | 80.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 80.49 | 80.73 | 80.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 80.55 | 80.69 | 80.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:30:00 | 80.44 | 80.69 | 80.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 80.42 | 80.64 | 80.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 82.54 | 80.64 | 80.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 79.34 | 80.74 | 80.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 79.34 | 80.74 | 80.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 76.22 | 78.79 | 79.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 76.60 | 76.11 | 77.55 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:00:00 | 74.78 | 75.73 | 76.91 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:30:00 | 74.48 | 75.41 | 76.66 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 76.47 | 75.73 | 76.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 76.47 | 75.73 | 76.12 | SL hit (close>ema400) qty=1.00 sl=76.12 alert=retest1 |

### Cycle 81 — BUY (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 13:15:00 | 76.46 | 74.85 | 74.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 15:15:00 | 76.68 | 75.45 | 75.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 75.43 | 75.65 | 75.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 11:30:00 | 75.52 | 75.65 | 75.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 75.53 | 75.62 | 75.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 75.32 | 75.62 | 75.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 75.39 | 75.58 | 75.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 75.39 | 75.58 | 75.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 74.90 | 75.44 | 75.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:45:00 | 74.76 | 75.44 | 75.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 74.63 | 75.28 | 75.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 73.67 | 75.28 | 75.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 73.81 | 74.99 | 75.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 71.60 | 74.31 | 74.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 71.96 | 71.50 | 72.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:00:00 | 71.96 | 71.50 | 72.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 72.47 | 71.69 | 72.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 72.42 | 71.69 | 72.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 72.37 | 71.83 | 72.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 72.89 | 71.83 | 72.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 73.61 | 72.18 | 72.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 73.61 | 72.18 | 72.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 72.63 | 72.32 | 72.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:00:00 | 72.63 | 72.32 | 72.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 72.67 | 72.39 | 72.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:45:00 | 72.66 | 72.39 | 72.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 72.79 | 72.47 | 72.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:00:00 | 72.79 | 72.47 | 72.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 73.37 | 72.65 | 72.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 73.37 | 72.65 | 72.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 15:15:00 | 73.40 | 72.80 | 72.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 73.88 | 73.02 | 72.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 14:15:00 | 74.20 | 74.27 | 73.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 15:00:00 | 74.20 | 74.27 | 73.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 74.69 | 74.93 | 74.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:30:00 | 74.99 | 74.88 | 74.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 13:15:00 | 74.28 | 74.72 | 74.56 | SL hit (close<static) qty=1.00 sl=74.43 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 73.97 | 74.44 | 74.45 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 13:15:00 | 75.22 | 74.58 | 74.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 76.48 | 74.96 | 74.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 75.93 | 76.29 | 75.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:00:00 | 75.93 | 76.29 | 75.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 75.70 | 76.17 | 75.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 75.60 | 76.17 | 75.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 75.93 | 76.13 | 75.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:30:00 | 75.70 | 76.13 | 75.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 75.82 | 76.05 | 75.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 13:30:00 | 76.00 | 76.05 | 75.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 75.35 | 75.91 | 75.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 75.35 | 75.91 | 75.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 75.43 | 75.82 | 75.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 75.43 | 75.82 | 75.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 74.51 | 75.48 | 75.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 74.14 | 75.21 | 75.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 11:15:00 | 74.70 | 74.69 | 75.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 13:15:00 | 74.57 | 74.70 | 74.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 13:45:00 | 74.50 | 74.67 | 74.94 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 14:30:00 | 74.43 | 74.58 | 74.87 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:15:00 | 70.84 | 71.85 | 72.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:15:00 | 70.77 | 71.65 | 72.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:15:00 | 70.71 | 71.65 | 72.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-05 13:15:00 | 70.79 | 70.78 | 71.34 | SL hit (close>ema200) qty=0.50 sl=70.78 alert=retest1 |

### Cycle 87 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 70.23 | 69.98 | 69.97 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 69.39 | 69.88 | 69.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 68.82 | 69.66 | 69.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 69.77 | 69.50 | 69.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 69.77 | 69.50 | 69.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 69.77 | 69.50 | 69.69 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 70.99 | 70.00 | 69.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 71.63 | 70.33 | 70.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 73.31 | 73.47 | 72.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:00:00 | 73.31 | 73.47 | 72.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 71.63 | 73.00 | 72.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 71.63 | 73.00 | 72.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 71.69 | 72.74 | 72.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 71.76 | 72.74 | 72.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 71.71 | 72.30 | 72.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 70.77 | 71.73 | 72.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 70.78 | 70.22 | 70.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 70.78 | 70.22 | 70.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 70.78 | 70.22 | 70.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 70.78 | 70.22 | 70.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 71.33 | 70.44 | 70.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 72.69 | 70.44 | 70.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 71.92 | 70.74 | 70.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 72.47 | 70.74 | 70.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 71.83 | 71.16 | 71.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 74.88 | 72.27 | 71.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 74.51 | 75.02 | 74.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 74.51 | 75.02 | 74.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 74.51 | 75.02 | 74.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:15:00 | 75.44 | 75.06 | 74.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 12:15:00 | 79.55 | 80.08 | 80.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 79.55 | 80.08 | 80.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 78.99 | 79.76 | 79.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 74.59 | 73.93 | 75.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 11:00:00 | 74.59 | 73.93 | 75.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 74.80 | 74.23 | 75.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:45:00 | 75.05 | 74.23 | 75.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 76.92 | 74.95 | 75.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 76.61 | 74.95 | 75.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 75.57 | 75.07 | 75.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 13:00:00 | 75.44 | 75.25 | 75.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 14:30:00 | 75.45 | 75.35 | 75.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 78.10 | 75.93 | 75.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 78.10 | 75.93 | 75.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 78.67 | 77.51 | 76.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 77.93 | 78.31 | 77.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 09:30:00 | 78.05 | 78.31 | 77.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 77.70 | 78.19 | 77.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 77.81 | 78.19 | 77.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 77.24 | 78.00 | 77.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 77.24 | 78.00 | 77.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 77.28 | 77.86 | 77.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 77.28 | 77.86 | 77.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 77.31 | 77.59 | 77.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 77.50 | 77.59 | 77.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 77.00 | 77.48 | 77.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:30:00 | 77.00 | 77.48 | 77.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 76.68 | 77.32 | 77.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 75.41 | 76.90 | 77.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 75.86 | 75.40 | 76.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:45:00 | 75.70 | 75.40 | 76.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 76.93 | 75.70 | 76.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 76.93 | 75.70 | 76.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 76.95 | 75.95 | 76.19 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 77.07 | 76.34 | 76.33 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 75.37 | 76.20 | 76.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 75.33 | 76.03 | 76.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 74.73 | 73.50 | 74.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 74.73 | 73.50 | 74.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 74.73 | 73.50 | 74.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:30:00 | 72.57 | 73.30 | 73.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 72.27 | 73.18 | 73.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 72.33 | 72.92 | 73.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:00:00 | 72.57 | 72.09 | 72.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 72.46 | 72.16 | 72.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 13:45:00 | 72.67 | 72.16 | 72.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 72.44 | 72.22 | 72.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:30:00 | 72.39 | 72.22 | 72.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 72.35 | 72.25 | 72.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 73.76 | 72.25 | 72.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 75.13 | 72.82 | 72.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 75.13 | 72.82 | 72.84 | SL hit (close>static) qty=1.00 sl=75.10 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 75.61 | 73.38 | 73.09 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 11:15:00 | 73.65 | 74.56 | 74.60 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 12:15:00 | 74.88 | 74.52 | 74.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 13:15:00 | 75.52 | 74.72 | 74.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 79.38 | 79.64 | 78.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 79.38 | 79.64 | 78.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 79.33 | 79.70 | 79.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 79.33 | 79.70 | 79.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 78.89 | 79.54 | 79.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 78.63 | 79.54 | 79.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 77.42 | 79.11 | 78.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 77.42 | 79.11 | 78.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 78.78 | 79.05 | 78.88 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 77.14 | 78.51 | 78.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 75.55 | 77.31 | 77.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 74.30 | 73.77 | 74.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 10:00:00 | 74.30 | 73.77 | 74.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 74.42 | 73.90 | 74.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:15:00 | 74.92 | 73.90 | 74.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 74.60 | 74.04 | 74.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:15:00 | 76.08 | 74.04 | 74.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 75.33 | 74.30 | 74.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 76.21 | 74.30 | 74.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 75.04 | 74.45 | 74.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:45:00 | 75.29 | 74.45 | 74.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 74.62 | 74.40 | 74.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 74.66 | 74.40 | 74.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 74.08 | 74.34 | 74.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:15:00 | 74.00 | 74.34 | 74.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:30:00 | 74.03 | 74.35 | 74.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 74.00 | 74.22 | 74.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 75.36 | 73.86 | 73.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 75.36 | 73.86 | 73.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 13:15:00 | 75.92 | 74.75 | 74.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 75.94 | 75.97 | 75.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:00:00 | 75.94 | 75.97 | 75.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 75.65 | 75.91 | 75.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 75.81 | 75.91 | 75.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 75.94 | 75.91 | 75.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 76.99 | 76.08 | 75.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 76.47 | 76.31 | 75.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 79.42 | 80.42 | 80.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 79.42 | 80.42 | 80.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 77.03 | 79.53 | 79.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 13:15:00 | 77.67 | 77.65 | 78.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:45:00 | 77.74 | 77.65 | 78.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 72.92 | 72.01 | 72.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 72.92 | 72.01 | 72.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 72.26 | 72.06 | 72.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 70.89 | 71.89 | 72.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 72.00 | 71.50 | 72.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:15:00 | 68.40 | 69.74 | 70.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 12:15:00 | 67.35 | 68.65 | 69.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-01 15:15:00 | 66.09 | 66.05 | 66.85 | SL hit (close>ema200) qty=0.50 sl=66.05 alert=retest2 |

### Cycle 103 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 67.74 | 67.04 | 67.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 11:15:00 | 68.37 | 67.49 | 67.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 67.66 | 67.68 | 67.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 67.66 | 67.68 | 67.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 66.63 | 67.47 | 67.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 66.15 | 67.47 | 67.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 65.60 | 67.09 | 67.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 65.30 | 66.33 | 66.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 65.75 | 65.66 | 66.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 65.75 | 65.66 | 66.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 66.06 | 65.73 | 66.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 65.92 | 65.73 | 66.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 66.37 | 65.86 | 66.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 66.37 | 65.86 | 66.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 65.62 | 65.81 | 66.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 64.96 | 65.81 | 66.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:00:00 | 65.28 | 65.70 | 65.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 61.71 | 63.38 | 64.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 62.02 | 63.38 | 64.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 61.75 | 61.34 | 62.59 | SL hit (close>ema200) qty=0.50 sl=61.34 alert=retest2 |

### Cycle 105 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 64.52 | 63.05 | 62.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 64.88 | 63.42 | 63.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 63.19 | 63.65 | 63.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 13:15:00 | 63.19 | 63.65 | 63.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 63.19 | 63.65 | 63.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:45:00 | 63.08 | 63.65 | 63.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 63.20 | 63.56 | 63.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 63.20 | 63.56 | 63.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 63.07 | 63.46 | 63.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 64.92 | 63.46 | 63.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 65.06 | 66.12 | 66.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 65.06 | 66.12 | 66.23 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 68.20 | 66.33 | 66.16 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 65.37 | 66.77 | 66.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 64.73 | 65.70 | 66.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 64.73 | 64.66 | 65.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 64.73 | 64.66 | 65.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 64.62 | 64.75 | 65.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 64.00 | 64.59 | 65.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 65.70 | 64.74 | 65.21 | SL hit (close>static) qty=1.00 sl=65.48 alert=retest2 |

### Cycle 109 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 65.90 | 65.37 | 65.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 66.18 | 65.53 | 65.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 64.95 | 65.46 | 65.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 11:15:00 | 64.95 | 65.46 | 65.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 64.95 | 65.46 | 65.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 64.95 | 65.46 | 65.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 64.92 | 65.35 | 65.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 64.50 | 65.18 | 65.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 65.75 | 65.23 | 65.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 10:15:00 | 65.75 | 65.23 | 65.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 65.75 | 65.23 | 65.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 65.75 | 65.23 | 65.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 65.68 | 65.32 | 65.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 66.04 | 65.53 | 65.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 65.56 | 65.77 | 65.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 65.56 | 65.77 | 65.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 65.56 | 65.77 | 65.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 65.50 | 65.77 | 65.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 64.67 | 65.55 | 65.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 64.67 | 65.55 | 65.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 65.13 | 65.47 | 65.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 62.10 | 64.48 | 65.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 63.72 | 63.01 | 63.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 63.72 | 63.01 | 63.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 63.72 | 63.01 | 63.79 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 65.05 | 63.99 | 63.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 65.25 | 64.24 | 64.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 64.36 | 64.88 | 64.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 11:15:00 | 64.36 | 64.88 | 64.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 64.36 | 64.88 | 64.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 64.36 | 64.88 | 64.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 65.37 | 64.98 | 64.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 64.75 | 64.98 | 64.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 64.90 | 64.96 | 64.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 64.75 | 64.96 | 64.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 64.85 | 64.94 | 64.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 64.89 | 64.94 | 64.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 64.83 | 65.96 | 65.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 64.83 | 65.96 | 65.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 64.71 | 65.71 | 65.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 64.60 | 65.71 | 65.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 64.32 | 65.19 | 65.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 62.50 | 64.21 | 64.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 62.67 | 62.00 | 62.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 12:15:00 | 62.67 | 62.00 | 62.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 62.67 | 62.00 | 62.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 62.67 | 62.00 | 62.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 62.61 | 62.13 | 62.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 63.15 | 62.13 | 62.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 62.87 | 62.27 | 62.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 62.87 | 62.27 | 62.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 63.05 | 62.43 | 62.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 62.89 | 62.43 | 62.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 64.15 | 62.77 | 62.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 64.15 | 62.77 | 62.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 64.10 | 63.04 | 63.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 64.10 | 63.04 | 63.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 64.04 | 63.24 | 63.17 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 62.08 | 63.02 | 63.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 61.40 | 62.49 | 62.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 61.98 | 61.95 | 62.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 61.98 | 61.95 | 62.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 62.21 | 62.01 | 62.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 62.07 | 62.01 | 62.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 62.67 | 62.14 | 62.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 62.67 | 62.14 | 62.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 62.46 | 62.20 | 62.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 61.49 | 62.20 | 62.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 62.44 | 61.99 | 62.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 62.44 | 61.99 | 62.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 63.13 | 62.22 | 62.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 63.13 | 62.22 | 62.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 62.95 | 62.36 | 62.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 64.33 | 62.76 | 62.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 66.18 | 66.96 | 66.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 66.18 | 66.96 | 66.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 66.18 | 66.96 | 66.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 66.18 | 66.96 | 66.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 66.31 | 66.83 | 66.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 66.21 | 66.83 | 66.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 66.11 | 66.69 | 66.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:00:00 | 66.11 | 66.69 | 66.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 66.13 | 66.58 | 66.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 66.02 | 66.58 | 66.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 65.64 | 66.39 | 66.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:30:00 | 65.58 | 66.39 | 66.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 66.15 | 66.34 | 66.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 66.34 | 66.19 | 66.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 11:15:00 | 65.19 | 65.90 | 65.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 65.19 | 65.90 | 65.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 64.58 | 65.17 | 65.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 65.20 | 64.79 | 65.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 65.20 | 64.79 | 65.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 65.20 | 64.79 | 65.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:30:00 | 64.89 | 64.79 | 65.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 65.01 | 64.83 | 65.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 63.27 | 64.83 | 65.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 12:15:00 | 64.60 | 63.34 | 63.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 64.60 | 63.34 | 63.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 65.40 | 63.75 | 63.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 67.01 | 67.15 | 66.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 67.01 | 67.15 | 66.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 66.09 | 67.04 | 66.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 66.09 | 67.04 | 66.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 65.59 | 66.75 | 66.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 64.60 | 66.75 | 66.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 64.68 | 66.34 | 66.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 63.68 | 64.84 | 65.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 64.85 | 64.71 | 65.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 64.85 | 64.71 | 65.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 64.76 | 64.72 | 65.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 64.24 | 64.66 | 65.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:15:00 | 64.41 | 64.67 | 65.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 64.16 | 64.57 | 64.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:00:00 | 64.58 | 64.33 | 64.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 64.67 | 64.40 | 64.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 64.80 | 64.40 | 64.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 64.70 | 64.46 | 64.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 64.77 | 64.46 | 64.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 64.49 | 64.47 | 64.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:15:00 | 64.83 | 64.47 | 64.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 65.01 | 64.57 | 64.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:30:00 | 64.95 | 64.57 | 64.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 64.80 | 64.62 | 64.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 65.06 | 64.62 | 64.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 65.13 | 64.72 | 64.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 65.71 | 64.72 | 64.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 66.10 | 65.00 | 64.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 66.10 | 65.00 | 64.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 66.90 | 66.22 | 65.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 67.18 | 68.30 | 67.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 67.18 | 68.30 | 67.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 67.18 | 68.30 | 67.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 15:00:00 | 67.57 | 67.58 | 67.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 68.17 | 67.58 | 67.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 67.72 | 68.35 | 68.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 67.72 | 68.35 | 68.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 67.50 | 67.95 | 68.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 68.08 | 67.97 | 68.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 68.08 | 67.97 | 68.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 68.08 | 67.97 | 68.17 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 68.62 | 68.22 | 68.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 68.88 | 68.35 | 68.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 68.97 | 69.30 | 69.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 68.97 | 69.30 | 69.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 68.97 | 69.30 | 69.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 70.20 | 69.66 | 69.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 15:00:00 | 70.22 | 69.77 | 69.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 70.65 | 70.04 | 69.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 70.48 | 70.19 | 69.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 66.93 | 69.59 | 69.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 66.93 | 69.59 | 69.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 66.74 | 69.02 | 69.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 61.95 | 61.92 | 63.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 61.95 | 61.92 | 63.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 62.45 | 61.31 | 61.98 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 62.93 | 62.41 | 62.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 64.38 | 62.90 | 62.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 65.31 | 65.36 | 64.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 65.31 | 65.36 | 64.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 66.51 | 67.63 | 67.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:30:00 | 67.20 | 67.63 | 67.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 66.82 | 67.47 | 67.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 66.60 | 67.47 | 67.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 68.05 | 68.13 | 67.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 68.27 | 68.13 | 67.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 66.12 | 67.73 | 67.68 | SL hit (close<static) qty=1.00 sl=67.80 alert=retest2 |

### Cycle 126 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 64.90 | 67.16 | 67.43 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 66.06 | 65.74 | 65.71 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 65.39 | 65.71 | 65.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 65.09 | 65.51 | 65.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 65.24 | 64.93 | 65.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 65.24 | 64.93 | 65.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 65.30 | 65.01 | 65.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 65.49 | 65.01 | 65.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 65.59 | 65.12 | 65.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 65.65 | 65.12 | 65.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 65.62 | 65.39 | 65.38 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 64.87 | 65.29 | 65.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 64.32 | 65.10 | 65.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 64.38 | 64.20 | 64.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 11:15:00 | 64.38 | 64.20 | 64.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 64.38 | 64.20 | 64.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:00:00 | 64.38 | 64.20 | 64.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 66.50 | 64.70 | 64.72 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 66.91 | 65.14 | 64.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 67.77 | 65.95 | 65.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 67.21 | 67.46 | 66.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 67.21 | 67.46 | 66.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 70.50 | 70.20 | 69.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 69.91 | 70.20 | 69.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 70.40 | 70.41 | 70.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 70.71 | 70.37 | 70.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 69.84 | 70.37 | 70.19 | SL hit (close<static) qty=1.00 sl=70.01 alert=retest2 |

### Cycle 132 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 69.45 | 70.04 | 70.06 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 70.14 | 70.02 | 70.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 70.41 | 70.10 | 70.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 72.40 | 72.82 | 72.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 72.40 | 72.82 | 72.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 72.40 | 72.82 | 72.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 72.40 | 72.82 | 72.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 72.27 | 72.66 | 72.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 72.27 | 72.66 | 72.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 72.87 | 72.70 | 72.39 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 71.09 | 72.12 | 72.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 70.74 | 71.84 | 72.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 71.92 | 71.48 | 71.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 71.92 | 71.48 | 71.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 71.92 | 71.48 | 71.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 71.92 | 71.48 | 71.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 71.51 | 71.49 | 71.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 71.70 | 71.49 | 71.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 71.48 | 71.49 | 71.74 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 72.62 | 71.93 | 71.88 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 70.79 | 71.70 | 71.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 12:15:00 | 70.57 | 71.22 | 71.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 71.55 | 70.94 | 71.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 12:15:00 | 71.55 | 70.94 | 71.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 71.55 | 70.94 | 71.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 71.55 | 70.94 | 71.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 71.20 | 71.00 | 71.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:15:00 | 70.97 | 71.00 | 71.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 71.19 | 70.96 | 71.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 71.85 | 71.14 | 71.19 | SL hit (close>static) qty=1.00 sl=71.69 alert=retest2 |

### Cycle 137 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 71.89 | 71.29 | 71.25 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 70.74 | 71.18 | 71.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 69.85 | 70.83 | 71.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 15:15:00 | 70.40 | 70.32 | 70.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 09:15:00 | 70.86 | 70.32 | 70.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 71.10 | 70.48 | 70.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 71.14 | 70.48 | 70.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 70.84 | 70.55 | 70.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 70.65 | 70.76 | 70.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 70.82 | 70.77 | 70.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 70.82 | 70.77 | 70.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 70.89 | 70.79 | 70.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 74.01 | 74.35 | 73.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 74.01 | 74.35 | 73.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 74.01 | 74.35 | 73.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 74.16 | 74.35 | 73.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 73.99 | 74.17 | 73.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 73.81 | 74.17 | 73.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 73.90 | 74.11 | 73.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 73.90 | 74.11 | 73.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 73.44 | 73.98 | 73.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 73.44 | 73.98 | 73.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 73.34 | 73.85 | 73.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 72.92 | 73.85 | 73.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 72.33 | 73.55 | 73.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 71.42 | 72.78 | 73.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 70.60 | 70.31 | 71.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 70.60 | 70.31 | 71.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 67.94 | 67.77 | 68.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 68.06 | 67.77 | 68.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 68.09 | 67.84 | 68.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 68.09 | 67.84 | 68.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 68.39 | 67.95 | 68.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 68.22 | 67.95 | 68.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 68.13 | 68.05 | 68.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 69.29 | 68.05 | 68.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 69.53 | 68.35 | 68.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 70.03 | 68.69 | 68.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 69.29 | 69.34 | 68.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 69.29 | 69.34 | 68.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 68.91 | 69.20 | 68.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 68.91 | 69.20 | 68.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 69.05 | 69.17 | 68.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:30:00 | 68.89 | 69.17 | 68.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 69.08 | 69.15 | 69.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 69.74 | 69.15 | 69.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 67.73 | 69.66 | 69.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 67.73 | 69.66 | 69.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 67.23 | 68.18 | 68.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 68.09 | 67.91 | 68.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 68.09 | 67.91 | 68.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 69.25 | 68.20 | 68.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 69.25 | 68.20 | 68.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 68.87 | 68.33 | 68.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 68.69 | 68.51 | 68.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 69.15 | 68.64 | 68.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 69.15 | 68.64 | 68.62 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 68.29 | 68.67 | 68.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 68.04 | 68.48 | 68.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 68.40 | 68.34 | 68.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 68.40 | 68.34 | 68.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 68.40 | 68.34 | 68.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 68.57 | 68.34 | 68.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 67.97 | 68.26 | 68.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 68.32 | 68.26 | 68.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 68.16 | 68.18 | 68.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 68.13 | 68.18 | 68.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 67.84 | 68.10 | 68.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 68.45 | 68.14 | 68.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 68.45 | 68.14 | 68.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 68.78 | 68.32 | 68.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 69.03 | 69.05 | 68.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 69.03 | 69.05 | 68.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 69.27 | 69.09 | 68.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 69.79 | 69.09 | 68.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 68.29 | 68.96 | 68.90 | SL hit (close<static) qty=1.00 sl=68.56 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 10:15:00 | 68.20 | 68.81 | 68.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 67.84 | 68.20 | 68.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 68.63 | 68.28 | 68.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 68.63 | 68.28 | 68.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 68.63 | 68.28 | 68.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 68.63 | 68.28 | 68.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 68.72 | 68.37 | 68.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 68.70 | 68.37 | 68.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 68.70 | 68.44 | 68.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 68.68 | 68.44 | 68.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 69.11 | 68.67 | 68.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 13:15:00 | 69.45 | 69.01 | 68.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 72.01 | 72.07 | 71.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:00:00 | 72.01 | 72.07 | 71.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 71.95 | 72.51 | 72.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 71.95 | 72.51 | 72.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 71.72 | 72.35 | 72.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 71.72 | 72.35 | 72.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 71.66 | 72.13 | 72.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 71.30 | 71.87 | 72.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 71.16 | 71.14 | 71.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 71.13 | 71.14 | 71.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 71.47 | 71.21 | 71.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 71.47 | 71.21 | 71.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 71.79 | 71.32 | 71.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 71.79 | 71.32 | 71.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 71.83 | 71.43 | 71.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 71.89 | 71.43 | 71.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 71.79 | 71.55 | 71.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 72.15 | 71.67 | 71.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 71.84 | 71.89 | 71.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 71.31 | 71.89 | 71.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 71.01 | 71.72 | 71.69 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 71.08 | 71.59 | 71.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 70.79 | 71.18 | 71.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 71.19 | 71.10 | 71.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 71.19 | 71.10 | 71.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 71.19 | 71.10 | 71.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 70.96 | 71.10 | 71.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:00:00 | 70.94 | 71.10 | 71.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 13:45:00 | 70.96 | 71.05 | 71.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 71.89 | 71.33 | 71.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 71.89 | 71.33 | 71.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 72.65 | 71.60 | 71.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 11:15:00 | 72.01 | 72.03 | 71.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:00:00 | 72.01 | 72.03 | 71.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 71.84 | 72.00 | 71.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 71.84 | 72.00 | 71.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 71.50 | 71.90 | 71.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 71.50 | 71.90 | 71.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 71.92 | 71.90 | 71.75 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 71.29 | 71.69 | 71.69 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 14:15:00 | 71.91 | 71.74 | 71.71 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 71.09 | 71.62 | 71.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 70.82 | 71.35 | 71.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 71.45 | 71.13 | 71.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 71.45 | 71.13 | 71.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 71.45 | 71.13 | 71.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 71.45 | 71.13 | 71.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 71.56 | 71.21 | 71.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 71.44 | 71.21 | 71.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 71.29 | 71.22 | 71.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 70.95 | 71.20 | 71.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 70.93 | 71.16 | 71.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:15:00 | 70.95 | 71.16 | 71.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 71.00 | 70.95 | 71.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 70.85 | 70.93 | 70.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 71.62 | 70.93 | 70.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 72.19 | 71.18 | 71.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 72.19 | 71.18 | 71.09 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 69.14 | 71.06 | 71.24 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 70.75 | 70.28 | 70.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 70.89 | 70.47 | 70.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 71.25 | 71.51 | 71.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:00:00 | 71.25 | 71.51 | 71.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 71.28 | 71.46 | 71.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 70.87 | 71.46 | 71.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 70.77 | 71.32 | 71.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 70.95 | 71.32 | 71.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 70.93 | 71.24 | 71.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 70.73 | 71.24 | 71.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 70.68 | 71.09 | 71.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 70.59 | 70.99 | 71.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 70.94 | 70.75 | 70.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 70.94 | 70.75 | 70.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 70.94 | 70.75 | 70.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 71.06 | 70.75 | 70.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 70.78 | 70.75 | 70.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 71.03 | 70.75 | 70.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 70.88 | 70.78 | 70.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:15:00 | 70.92 | 70.78 | 70.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 70.71 | 70.77 | 70.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 69.86 | 70.74 | 70.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 70.97 | 69.70 | 69.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 70.97 | 69.70 | 69.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 72.02 | 70.17 | 69.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 73.57 | 73.65 | 72.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 73.35 | 73.51 | 73.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 73.35 | 73.51 | 73.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 12:45:00 | 73.47 | 73.42 | 73.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 75.32 | 75.75 | 75.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 75.32 | 75.75 | 75.75 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 76.24 | 75.78 | 75.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 76.40 | 75.94 | 75.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 13:15:00 | 76.72 | 76.78 | 76.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 76.72 | 76.78 | 76.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 76.58 | 76.74 | 76.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 76.44 | 76.74 | 76.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 76.65 | 76.72 | 76.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 76.89 | 76.72 | 76.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 15:00:00 | 76.83 | 77.02 | 76.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 76.84 | 77.29 | 77.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 76.84 | 77.29 | 77.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 76.34 | 77.05 | 77.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 75.84 | 75.76 | 76.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 75.90 | 75.76 | 76.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 75.42 | 75.28 | 75.74 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 76.55 | 75.91 | 75.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 77.52 | 76.56 | 76.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 76.68 | 76.86 | 76.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 76.68 | 76.86 | 76.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 76.68 | 76.86 | 76.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 76.49 | 76.86 | 76.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 76.04 | 76.70 | 76.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 76.04 | 76.70 | 76.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 76.11 | 76.58 | 76.45 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 76.00 | 76.30 | 76.33 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 76.85 | 76.29 | 76.28 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 76.03 | 76.24 | 76.26 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 76.58 | 76.31 | 76.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 09:15:00 | 77.89 | 76.60 | 76.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 77.17 | 77.90 | 77.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 77.17 | 77.90 | 77.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 77.17 | 77.90 | 77.35 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 76.55 | 77.10 | 77.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 76.05 | 76.83 | 76.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 76.60 | 76.35 | 76.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 76.60 | 76.35 | 76.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 76.60 | 76.35 | 76.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 76.60 | 76.35 | 76.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 76.72 | 76.42 | 76.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 76.72 | 76.42 | 76.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 76.80 | 76.50 | 76.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 76.82 | 76.50 | 76.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 76.70 | 76.62 | 76.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 76.20 | 76.62 | 76.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 74.68 | 74.58 | 74.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 74.68 | 74.58 | 74.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 76.68 | 75.01 | 74.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 76.12 | 76.24 | 75.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 09:45:00 | 75.95 | 76.24 | 75.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 75.72 | 76.04 | 75.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 75.75 | 76.04 | 75.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 75.87 | 76.00 | 75.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:30:00 | 76.06 | 75.82 | 75.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 76.56 | 75.79 | 75.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 74.50 | 75.73 | 75.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 74.50 | 75.73 | 75.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 74.18 | 74.91 | 75.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 73.42 | 73.30 | 73.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 73.54 | 73.30 | 73.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 74.13 | 73.46 | 73.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 74.13 | 73.46 | 73.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 73.97 | 73.56 | 73.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 74.07 | 73.56 | 73.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 74.22 | 73.70 | 73.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 74.32 | 73.70 | 73.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 74.18 | 73.79 | 73.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 75.79 | 73.79 | 73.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 76.32 | 74.30 | 74.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 76.65 | 75.63 | 75.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 77.15 | 77.40 | 76.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:15:00 | 77.20 | 77.40 | 76.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 76.90 | 77.30 | 76.78 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 76.52 | 76.63 | 76.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 75.01 | 76.31 | 76.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 75.94 | 75.87 | 76.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:45:00 | 75.86 | 75.87 | 76.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 73.72 | 72.87 | 73.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 73.99 | 72.87 | 73.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 73.83 | 73.06 | 73.22 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 73.88 | 73.36 | 73.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 74.32 | 73.66 | 73.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 73.93 | 74.07 | 73.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 73.93 | 74.07 | 73.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 74.15 | 74.09 | 73.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 74.60 | 73.93 | 73.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 74.52 | 74.10 | 73.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 75.51 | 75.91 | 75.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 75.51 | 75.91 | 75.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 75.14 | 75.76 | 75.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 75.06 | 74.81 | 75.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 13:15:00 | 75.06 | 74.81 | 75.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 75.06 | 74.81 | 75.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 75.10 | 74.81 | 75.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 75.07 | 74.87 | 75.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 75.33 | 74.87 | 75.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 74.98 | 74.89 | 75.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 74.40 | 74.77 | 74.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 75.32 | 74.93 | 74.96 | SL hit (close>static) qty=1.00 sl=75.30 alert=retest2 |

### Cycle 175 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 75.25 | 74.99 | 74.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 76.06 | 75.32 | 75.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 77.52 | 77.86 | 77.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:30:00 | 77.44 | 77.86 | 77.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 76.90 | 77.58 | 77.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 76.90 | 77.58 | 77.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 76.90 | 77.45 | 77.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 76.90 | 77.45 | 77.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 76.81 | 77.32 | 77.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:30:00 | 76.88 | 77.32 | 77.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 77.20 | 77.30 | 77.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 77.46 | 77.24 | 77.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 77.55 | 77.24 | 77.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 76.48 | 77.03 | 77.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 76.48 | 77.03 | 77.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 76.10 | 76.70 | 76.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 76.29 | 76.23 | 76.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 76.29 | 76.23 | 76.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 77.55 | 76.50 | 76.63 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 78.06 | 76.81 | 76.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 78.34 | 77.31 | 77.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 81.23 | 81.34 | 80.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:15:00 | 82.29 | 81.34 | 80.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 81.12 | 82.08 | 81.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 81.12 | 82.08 | 81.65 | SL hit (close<ema400) qty=1.00 sl=81.65 alert=retest1 |

### Cycle 178 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 82.83 | 84.20 | 84.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 82.11 | 83.55 | 83.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 81.29 | 80.77 | 81.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 81.29 | 80.77 | 81.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 81.79 | 81.02 | 81.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 81.79 | 81.02 | 81.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 81.93 | 81.20 | 81.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 81.58 | 81.20 | 81.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 82.27 | 81.56 | 81.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 82.10 | 81.56 | 81.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 81.70 | 81.59 | 81.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 81.83 | 81.59 | 81.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 81.39 | 81.55 | 81.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 81.64 | 81.55 | 81.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 82.10 | 81.66 | 81.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 82.10 | 81.66 | 81.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 82.05 | 81.74 | 81.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 83.02 | 81.74 | 81.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 83.13 | 82.02 | 81.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 83.84 | 82.38 | 82.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 83.29 | 83.30 | 82.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 09:30:00 | 83.02 | 83.30 | 82.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 82.78 | 83.10 | 82.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 82.78 | 83.10 | 82.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 82.84 | 83.05 | 82.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 83.00 | 83.05 | 82.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 82.48 | 82.94 | 82.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 82.01 | 82.94 | 82.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 81.75 | 82.70 | 82.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 81.40 | 82.44 | 82.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 78.88 | 78.71 | 79.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 80.00 | 78.71 | 79.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 79.01 | 78.77 | 79.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 80.15 | 78.77 | 79.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 78.66 | 77.55 | 78.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 78.66 | 77.55 | 78.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 78.33 | 77.70 | 78.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:30:00 | 77.82 | 77.71 | 78.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 78.73 | 77.92 | 78.07 | SL hit (close>static) qty=1.00 sl=78.72 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 79.30 | 78.19 | 78.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 81.32 | 78.93 | 78.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 81.07 | 82.99 | 81.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 81.07 | 82.99 | 81.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 81.07 | 82.99 | 81.75 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 80.49 | 81.28 | 81.30 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 82.40 | 81.50 | 81.40 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 80.50 | 81.36 | 81.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 80.29 | 81.14 | 81.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 80.62 | 80.24 | 80.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 13:15:00 | 80.62 | 80.24 | 80.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 80.62 | 80.24 | 80.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 80.54 | 80.24 | 80.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 81.42 | 80.48 | 80.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 81.42 | 80.48 | 80.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 81.90 | 80.76 | 80.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 81.90 | 80.76 | 80.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 81.65 | 81.05 | 80.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 82.13 | 81.27 | 81.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 81.69 | 81.72 | 81.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 14:15:00 | 81.69 | 81.72 | 81.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 81.69 | 81.72 | 81.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 83.70 | 81.66 | 81.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 80.63 | 84.09 | 84.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 80.63 | 84.09 | 84.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 80.02 | 81.93 | 83.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 80.41 | 80.14 | 81.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 80.41 | 80.14 | 81.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 79.93 | 79.42 | 80.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 80.12 | 79.42 | 80.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 80.03 | 79.63 | 80.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 80.03 | 79.63 | 80.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 80.07 | 79.72 | 80.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 80.07 | 79.72 | 80.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 80.09 | 79.80 | 80.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 80.57 | 79.80 | 80.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 79.89 | 79.81 | 80.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:00:00 | 79.37 | 79.83 | 79.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 79.39 | 79.54 | 79.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 79.44 | 79.18 | 79.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 79.64 | 79.40 | 79.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 79.64 | 79.40 | 79.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 80.65 | 79.70 | 79.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 82.03 | 82.23 | 81.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 81.79 | 82.15 | 81.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 81.79 | 82.15 | 81.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 81.79 | 82.15 | 81.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 81.44 | 82.01 | 81.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 81.05 | 82.01 | 81.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 81.65 | 81.94 | 81.76 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 80.55 | 81.54 | 81.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 78.20 | 80.65 | 81.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 78.75 | 78.03 | 79.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 78.75 | 78.03 | 79.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 79.12 | 78.25 | 79.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 79.18 | 78.25 | 79.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 78.04 | 78.21 | 79.12 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 80.13 | 79.36 | 79.28 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 78.54 | 79.38 | 79.39 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 79.71 | 79.25 | 79.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 80.53 | 79.72 | 79.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 80.00 | 80.01 | 79.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 80.00 | 80.01 | 79.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 79.62 | 79.94 | 79.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 79.62 | 79.94 | 79.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 79.57 | 79.87 | 79.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 78.33 | 79.87 | 79.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 78.90 | 79.67 | 79.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 79.58 | 79.77 | 79.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 79.34 | 80.28 | 80.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 79.47 | 80.02 | 79.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 79.55 | 80.02 | 79.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 79.29 | 79.88 | 79.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 79.29 | 79.88 | 79.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 78.77 | 79.66 | 79.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 78.15 | 77.82 | 78.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 78.15 | 77.82 | 78.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 78.15 | 77.82 | 78.45 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 79.25 | 78.62 | 78.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 79.63 | 78.91 | 78.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 78.02 | 78.99 | 78.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 78.02 | 78.99 | 78.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 78.02 | 78.99 | 78.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 78.75 | 78.88 | 78.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 78.24 | 78.71 | 78.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 78.24 | 78.71 | 78.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 77.90 | 78.55 | 78.66 | Break + close below crossover candle low |

### Cycle 195 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 80.19 | 78.70 | 78.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 80.97 | 79.66 | 79.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 79.74 | 79.82 | 79.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 79.74 | 79.82 | 79.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 196 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 75.45 | 78.90 | 79.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 75.19 | 78.16 | 78.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 76.50 | 75.90 | 76.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 76.50 | 75.90 | 76.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 77.04 | 76.13 | 76.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 76.91 | 76.13 | 76.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 76.75 | 76.25 | 76.81 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 77.97 | 77.20 | 77.13 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 76.92 | 77.36 | 77.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 76.22 | 77.13 | 77.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 78.76 | 77.33 | 77.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 78.76 | 77.33 | 77.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 78.76 | 77.33 | 77.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 79.10 | 77.33 | 77.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 78.34 | 77.53 | 77.44 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 75.91 | 77.35 | 77.49 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 77.95 | 77.51 | 77.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 79.96 | 78.00 | 77.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 11:15:00 | 84.22 | 84.25 | 83.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:30:00 | 84.13 | 84.25 | 83.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 83.92 | 84.44 | 83.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 84.63 | 84.44 | 83.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 87.00 | 88.34 | 88.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 87.00 | 88.34 | 88.47 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 13:15:00 | 89.18 | 88.07 | 88.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 89.27 | 88.31 | 88.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 90.71 | 90.75 | 90.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 90.71 | 90.75 | 90.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 90.86 | 90.88 | 90.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 91.99 | 91.15 | 90.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:45:00 | 91.67 | 91.34 | 90.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 14:30:00 | 92.01 | 91.49 | 90.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 90.37 | 90.65 | 90.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 90.37 | 90.65 | 90.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 89.84 | 90.34 | 90.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 89.72 | 89.11 | 89.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 89.72 | 89.11 | 89.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 89.72 | 89.11 | 89.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 89.43 | 89.08 | 89.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 90.67 | 89.39 | 89.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 90.67 | 89.39 | 89.39 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 89.05 | 89.63 | 89.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 88.80 | 89.36 | 89.51 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 80.72 | 2024-04-18 14:15:00 | 78.12 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-04-23 09:30:00 | 78.60 | 2024-04-24 10:15:00 | 80.32 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-04-23 11:00:00 | 78.80 | 2024-04-24 10:15:00 | 80.32 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-04-23 12:45:00 | 78.82 | 2024-04-24 10:15:00 | 80.32 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-04-23 14:15:00 | 78.67 | 2024-04-24 10:15:00 | 80.32 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-05-02 12:15:00 | 86.68 | 2024-05-07 14:15:00 | 86.97 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-05-02 13:15:00 | 86.35 | 2024-05-07 14:15:00 | 86.97 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-05-07 13:15:00 | 86.90 | 2024-05-07 14:15:00 | 86.97 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2024-05-13 09:15:00 | 83.87 | 2024-05-13 15:15:00 | 85.37 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-17 09:15:00 | 89.38 | 2024-05-22 14:15:00 | 91.02 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2024-05-27 09:30:00 | 88.45 | 2024-05-30 09:15:00 | 84.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:30:00 | 88.45 | 2024-05-31 10:15:00 | 84.55 | STOP_HIT | 0.50 | 4.41% |
| BUY | retest2 | 2024-06-11 09:15:00 | 85.82 | 2024-06-19 12:15:00 | 87.78 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2024-06-12 09:15:00 | 86.35 | 2024-06-19 12:15:00 | 87.78 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2024-07-01 12:15:00 | 83.21 | 2024-07-01 14:15:00 | 83.93 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-07-01 12:45:00 | 83.24 | 2024-07-01 14:15:00 | 83.93 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-09 09:30:00 | 84.48 | 2024-07-10 09:15:00 | 81.60 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-07-12 11:15:00 | 82.36 | 2024-07-15 15:15:00 | 82.48 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-07-23 10:15:00 | 76.99 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-07-23 11:15:00 | 76.95 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-07-23 11:45:00 | 77.11 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-07-23 12:15:00 | 74.76 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest2 | 2024-07-24 12:30:00 | 77.69 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-07-24 13:45:00 | 77.83 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-07-25 11:30:00 | 77.70 | 2024-07-26 09:15:00 | 78.94 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-08-01 09:15:00 | 82.54 | 2024-08-02 09:15:00 | 79.34 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest1 | 2024-08-06 14:00:00 | 74.78 | 2024-08-07 15:15:00 | 76.47 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest1 | 2024-08-06 14:30:00 | 74.48 | 2024-08-07 15:15:00 | 76.47 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-08-08 13:15:00 | 74.74 | 2024-08-12 13:15:00 | 76.46 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-08-08 13:45:00 | 74.82 | 2024-08-12 13:15:00 | 76.46 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-08-23 11:30:00 | 74.99 | 2024-08-23 13:15:00 | 74.28 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2024-08-30 13:15:00 | 74.57 | 2024-09-04 09:15:00 | 70.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-30 13:45:00 | 74.50 | 2024-09-04 10:15:00 | 70.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-30 14:30:00 | 74.43 | 2024-09-04 10:15:00 | 70.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-30 13:15:00 | 74.57 | 2024-09-05 13:15:00 | 70.79 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest1 | 2024-08-30 13:45:00 | 74.50 | 2024-09-05 13:15:00 | 70.79 | STOP_HIT | 0.50 | 4.98% |
| SELL | retest1 | 2024-08-30 14:30:00 | 74.43 | 2024-09-05 13:15:00 | 70.79 | STOP_HIT | 0.50 | 4.89% |
| BUY | retest2 | 2024-09-26 11:15:00 | 75.44 | 2024-10-04 12:15:00 | 79.55 | STOP_HIT | 1.00 | 5.45% |
| SELL | retest2 | 2024-10-10 13:00:00 | 75.44 | 2024-10-11 09:15:00 | 78.10 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-10-10 14:30:00 | 75.45 | 2024-10-11 09:15:00 | 78.10 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2024-10-23 14:30:00 | 72.57 | 2024-10-28 09:15:00 | 75.13 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-10-24 09:15:00 | 72.27 | 2024-10-28 09:15:00 | 75.13 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-10-24 12:00:00 | 72.33 | 2024-10-28 09:15:00 | 75.13 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-10-25 13:00:00 | 72.57 | 2024-10-28 09:15:00 | 75.13 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-11-19 11:15:00 | 74.00 | 2024-11-25 09:15:00 | 75.36 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-11-19 12:30:00 | 74.03 | 2024-11-25 09:15:00 | 75.36 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-11-19 14:45:00 | 74.00 | 2024-11-25 09:15:00 | 75.36 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-11-29 09:45:00 | 76.99 | 2024-12-12 10:15:00 | 79.42 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2024-11-29 13:15:00 | 76.47 | 2024-12-12 10:15:00 | 79.42 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2024-12-20 15:00:00 | 70.89 | 2024-12-30 09:15:00 | 68.40 | PARTIAL | 0.50 | 3.51% |
| SELL | retest2 | 2024-12-23 11:45:00 | 72.00 | 2024-12-30 12:15:00 | 67.35 | PARTIAL | 0.50 | 6.46% |
| SELL | retest2 | 2024-12-20 15:00:00 | 70.89 | 2025-01-01 15:15:00 | 66.09 | STOP_HIT | 0.50 | 6.77% |
| SELL | retest2 | 2024-12-23 11:45:00 | 72.00 | 2025-01-01 15:15:00 | 66.09 | STOP_HIT | 0.50 | 8.21% |
| SELL | retest2 | 2025-01-08 09:15:00 | 64.96 | 2025-01-13 09:15:00 | 61.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:00:00 | 65.28 | 2025-01-13 09:15:00 | 62.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 64.96 | 2025-01-14 09:15:00 | 61.75 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-01-09 11:00:00 | 65.28 | 2025-01-14 09:15:00 | 61.75 | STOP_HIT | 0.50 | 5.41% |
| BUY | retest2 | 2025-01-16 09:15:00 | 64.92 | 2025-01-22 10:15:00 | 65.06 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-01-28 14:45:00 | 64.00 | 2025-01-29 09:15:00 | 65.70 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-02-25 09:30:00 | 66.34 | 2025-02-25 11:15:00 | 65.19 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-02-28 09:15:00 | 63.27 | 2025-03-05 12:15:00 | 64.60 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-03-13 10:45:00 | 64.24 | 2025-03-18 09:15:00 | 66.10 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-03-13 12:15:00 | 64.41 | 2025-03-18 09:15:00 | 66.10 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-03-13 13:00:00 | 64.16 | 2025-03-18 09:15:00 | 66.10 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-03-17 10:00:00 | 64.58 | 2025-03-18 09:15:00 | 66.10 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-03-21 15:00:00 | 67.57 | 2025-03-26 11:15:00 | 67.72 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-03-24 09:15:00 | 68.17 | 2025-03-26 11:15:00 | 67.72 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-04-02 14:15:00 | 70.20 | 2025-04-04 09:15:00 | 66.93 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2025-04-02 15:00:00 | 70.22 | 2025-04-04 09:15:00 | 66.93 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2025-04-03 11:30:00 | 70.65 | 2025-04-04 09:15:00 | 66.93 | STOP_HIT | 1.00 | -5.27% |
| BUY | retest2 | 2025-04-03 15:00:00 | 70.48 | 2025-04-04 09:15:00 | 66.93 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest2 | 2025-04-25 09:15:00 | 68.27 | 2025-04-25 09:15:00 | 66.12 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-05-20 09:15:00 | 70.71 | 2025-05-20 13:15:00 | 69.84 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-02 14:15:00 | 70.97 | 2025-06-03 10:15:00 | 71.85 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-03 09:30:00 | 71.19 | 2025-06-03 10:15:00 | 71.85 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-06-05 13:30:00 | 70.65 | 2025-06-05 14:15:00 | 70.82 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-06-26 09:15:00 | 69.74 | 2025-07-01 09:15:00 | 67.73 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-07-03 11:30:00 | 68.69 | 2025-07-03 12:15:00 | 69.15 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-08 14:15:00 | 68.13 | 2025-07-10 12:15:00 | 68.45 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-09 09:30:00 | 67.84 | 2025-07-10 12:15:00 | 68.45 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-14 10:15:00 | 69.79 | 2025-07-15 09:15:00 | 68.29 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-01 11:30:00 | 70.96 | 2025-08-04 11:15:00 | 71.89 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-01 12:00:00 | 70.94 | 2025-08-04 11:15:00 | 71.89 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-01 13:45:00 | 70.96 | 2025-08-04 11:15:00 | 71.89 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-08 12:15:00 | 70.95 | 2025-08-13 09:15:00 | 72.19 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-08-08 15:15:00 | 70.93 | 2025-08-13 09:15:00 | 72.19 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-11 11:15:00 | 70.95 | 2025-08-13 09:15:00 | 72.19 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-08-12 14:00:00 | 71.00 | 2025-08-13 09:15:00 | 72.19 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-08-26 09:15:00 | 69.86 | 2025-09-02 09:15:00 | 70.97 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-09-05 12:45:00 | 73.47 | 2025-09-16 09:15:00 | 75.32 | STOP_HIT | 1.00 | 2.52% |
| BUY | retest2 | 2025-09-22 09:15:00 | 76.89 | 2025-09-25 12:15:00 | 76.84 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-09-22 15:00:00 | 76.83 | 2025-09-25 12:15:00 | 76.84 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-10-16 09:15:00 | 76.20 | 2025-10-28 14:15:00 | 74.68 | STOP_HIT | 1.00 | 1.99% |
| BUY | retest2 | 2025-10-31 14:30:00 | 76.06 | 2025-11-04 09:15:00 | 74.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-11-03 09:15:00 | 76.56 | 2025-11-04 09:15:00 | 74.50 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-12-01 09:15:00 | 74.60 | 2025-12-08 10:15:00 | 75.51 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-12-01 12:00:00 | 74.52 | 2025-12-08 10:15:00 | 75.51 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2025-12-10 14:00:00 | 74.40 | 2025-12-11 11:15:00 | 75.32 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-17 09:45:00 | 77.46 | 2025-12-18 13:15:00 | 76.48 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-17 10:15:00 | 77.55 | 2025-12-18 13:15:00 | 76.48 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest1 | 2025-12-26 09:15:00 | 82.29 | 2025-12-29 12:15:00 | 81.12 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-01-01 11:15:00 | 83.10 | 2026-01-08 11:15:00 | 82.83 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-01-27 13:30:00 | 77.82 | 2026-01-27 14:15:00 | 78.73 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-02-04 09:15:00 | 83.70 | 2026-02-13 09:15:00 | 80.63 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-02-19 15:00:00 | 79.37 | 2026-02-24 12:15:00 | 79.64 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-02-23 09:45:00 | 79.39 | 2026-02-24 12:15:00 | 79.64 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-02-24 10:45:00 | 79.44 | 2026-02-24 12:15:00 | 79.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-03-12 10:45:00 | 79.58 | 2026-03-13 11:15:00 | 79.29 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2026-03-13 10:15:00 | 79.34 | 2026-03-13 11:15:00 | 79.29 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-03-13 10:45:00 | 79.47 | 2026-03-13 11:15:00 | 79.29 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-03-13 11:15:00 | 79.55 | 2026-03-13 11:15:00 | 79.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-03-19 10:30:00 | 78.75 | 2026-03-19 12:15:00 | 78.24 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-04-13 10:15:00 | 84.63 | 2026-04-23 09:15:00 | 87.00 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2026-04-29 10:45:00 | 91.99 | 2026-04-30 13:15:00 | 90.37 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-04-29 12:45:00 | 91.67 | 2026-04-30 13:15:00 | 90.37 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-29 14:30:00 | 92.01 | 2026-04-30 13:15:00 | 90.37 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-05-06 10:30:00 | 89.43 | 2026-05-07 09:15:00 | 90.67 | STOP_HIT | 1.00 | -1.39% |
