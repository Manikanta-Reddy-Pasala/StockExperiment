# Transformers And Rectifiers (India) Ltd. (TARIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5317 bars)
- **Last close:** 325.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 223 |
| ALERT1 | 149 |
| ALERT2 | 147 |
| ALERT2_SKIP | 100 |
| ALERT3 | 330 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 83 |
| PARTIAL | 20 |
| TARGET_HIT | 12 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 61
- **Target hits / Stop hits / Partials:** 12 / 77 / 20
- **Avg / median % per leg:** 0.93% / -0.57%
- **Sum % (uncompounded):** 100.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 9 | 23.7% | 8 | 29 | 1 | 0.23% | 8.8% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.76% | 8.8% |
| BUY @ 3rd Alert (retest2) | 33 | 7 | 21.2% | 7 | 26 | 0 | -0.00% | -0.0% |
| SELL (all) | 71 | 39 | 54.9% | 4 | 48 | 19 | 1.30% | 92.2% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.73% | -7.5% |
| SELL @ 3rd Alert (retest2) | 69 | 39 | 56.5% | 4 | 46 | 19 | 1.44% | 99.6% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.19% | 1.4% |
| retest2 (combined) | 102 | 46 | 45.1% | 11 | 72 | 19 | 0.98% | 99.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 33.25 | 32.50 | 32.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 12:15:00 | 33.38 | 32.80 | 32.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 11:15:00 | 34.63 | 34.86 | 34.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 34.75 | 34.89 | 34.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 34.75 | 34.89 | 34.55 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 10:15:00 | 34.25 | 34.44 | 34.46 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 15:15:00 | 34.75 | 34.44 | 34.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 37.20 | 34.99 | 34.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 37.85 | 37.97 | 36.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 14:15:00 | 38.88 | 39.10 | 38.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 38.88 | 39.10 | 38.62 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 14:15:00 | 40.33 | 41.52 | 41.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 15:15:00 | 40.00 | 41.22 | 41.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 11:15:00 | 41.40 | 41.16 | 41.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 11:15:00 | 41.40 | 41.16 | 41.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 41.40 | 41.16 | 41.38 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 41.20 | 40.07 | 40.01 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 14:15:00 | 40.33 | 40.70 | 40.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 09:15:00 | 40.03 | 40.48 | 40.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 40.00 | 39.92 | 40.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 40.00 | 39.92 | 40.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 40.00 | 39.92 | 40.20 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 43.20 | 40.60 | 40.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 10:15:00 | 43.43 | 41.17 | 40.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 14:15:00 | 44.35 | 44.61 | 43.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 43.68 | 44.36 | 43.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 43.68 | 44.36 | 43.93 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 42.85 | 43.71 | 43.73 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 11:15:00 | 44.43 | 43.75 | 43.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 12:15:00 | 44.85 | 43.97 | 43.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-26 09:15:00 | 43.80 | 44.16 | 44.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 43.80 | 44.16 | 44.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 43.80 | 44.16 | 44.00 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 14:15:00 | 48.15 | 49.44 | 49.45 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 49.83 | 49.47 | 49.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 50.65 | 49.80 | 49.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 49.90 | 50.26 | 50.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 49.90 | 50.26 | 50.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 49.90 | 50.26 | 50.02 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 49.10 | 49.74 | 49.82 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 50.65 | 49.94 | 49.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 54.63 | 51.13 | 50.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 12:15:00 | 53.98 | 53.99 | 52.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 52.73 | 53.85 | 53.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 52.73 | 53.85 | 53.21 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 13:15:00 | 51.70 | 52.73 | 52.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 14:15:00 | 51.23 | 52.43 | 52.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 14:15:00 | 52.05 | 52.01 | 52.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 15:15:00 | 52.13 | 52.03 | 52.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 15:15:00 | 52.13 | 52.03 | 52.28 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 53.85 | 52.58 | 52.48 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 09:15:00 | 45.73 | 51.65 | 52.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 13:15:00 | 42.85 | 47.01 | 49.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 43.50 | 41.89 | 44.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 10:15:00 | 44.58 | 42.43 | 44.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 44.58 | 42.43 | 44.38 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 45.45 | 44.18 | 44.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 09:15:00 | 45.63 | 45.00 | 44.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 13:15:00 | 44.23 | 45.03 | 44.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 13:15:00 | 44.23 | 45.03 | 44.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 44.23 | 45.03 | 44.73 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 12:15:00 | 44.20 | 44.56 | 44.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 15:15:00 | 43.75 | 44.23 | 44.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 10:15:00 | 44.25 | 44.22 | 44.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 43.88 | 43.91 | 44.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 43.88 | 43.91 | 44.12 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 45.90 | 43.87 | 43.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 11:15:00 | 46.88 | 46.16 | 45.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 47.88 | 51.87 | 50.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 47.88 | 51.87 | 50.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 47.88 | 51.87 | 50.89 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 12:15:00 | 47.83 | 49.96 | 50.15 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 14:15:00 | 53.18 | 50.69 | 50.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 09:15:00 | 53.70 | 51.64 | 50.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 14:15:00 | 55.00 | 55.03 | 54.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 53.90 | 54.84 | 54.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 53.90 | 54.84 | 54.22 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 14:15:00 | 52.95 | 53.85 | 53.92 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 54.80 | 53.88 | 53.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 57.93 | 54.79 | 54.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 60.23 | 60.30 | 59.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 10:15:00 | 58.98 | 60.03 | 59.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 58.98 | 60.03 | 59.11 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 14:15:00 | 59.50 | 60.32 | 60.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 15:15:00 | 56.25 | 59.24 | 59.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 12:15:00 | 58.73 | 58.61 | 59.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 13:15:00 | 58.53 | 58.60 | 59.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 58.53 | 58.60 | 59.19 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 12:15:00 | 60.10 | 59.43 | 59.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 13:15:00 | 60.48 | 59.64 | 59.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 09:15:00 | 59.63 | 60.00 | 59.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 59.63 | 60.00 | 59.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 59.63 | 60.00 | 59.71 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 10:15:00 | 58.50 | 59.55 | 59.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 11:15:00 | 57.90 | 59.22 | 59.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 11:15:00 | 58.60 | 58.59 | 58.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 12:15:00 | 59.35 | 58.75 | 59.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 59.35 | 58.75 | 59.01 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 61.30 | 59.26 | 59.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 14:15:00 | 62.75 | 59.96 | 59.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 74.30 | 75.02 | 71.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 74.30 | 75.02 | 71.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 74.30 | 75.02 | 71.41 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 10:15:00 | 80.63 | 83.22 | 83.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 78.53 | 81.00 | 82.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 81.98 | 80.81 | 81.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 10:15:00 | 81.98 | 80.81 | 81.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 81.98 | 80.81 | 81.76 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 82.65 | 81.51 | 81.39 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 14:15:00 | 81.30 | 81.65 | 81.67 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 82.75 | 81.77 | 81.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 11:15:00 | 84.70 | 82.51 | 82.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 83.98 | 84.90 | 83.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 83.98 | 84.90 | 83.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 83.98 | 84.90 | 83.99 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 82.43 | 83.42 | 83.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 81.65 | 82.85 | 83.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 83.08 | 80.88 | 81.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 83.08 | 80.88 | 81.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 83.08 | 80.88 | 81.73 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 13:15:00 | 84.65 | 82.60 | 82.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 15:15:00 | 86.35 | 83.85 | 82.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 14:15:00 | 86.40 | 86.45 | 84.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 83.68 | 85.79 | 84.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 83.68 | 85.79 | 84.87 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 82.98 | 84.15 | 84.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 82.63 | 83.85 | 84.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 84.70 | 84.02 | 84.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 84.70 | 84.02 | 84.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 84.70 | 84.02 | 84.21 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 84.88 | 84.28 | 84.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 85.13 | 84.55 | 84.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 14:15:00 | 84.35 | 84.61 | 84.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 14:15:00 | 84.35 | 84.61 | 84.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 84.35 | 84.61 | 84.48 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 14:15:00 | 85.30 | 85.97 | 86.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 84.00 | 85.57 | 85.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 74.75 | 74.44 | 76.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 11:15:00 | 75.85 | 74.48 | 75.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 11:15:00 | 75.85 | 74.48 | 75.99 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 15:15:00 | 79.58 | 76.98 | 76.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 09:15:00 | 81.00 | 77.78 | 77.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 83.55 | 84.36 | 82.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 83.43 | 84.24 | 83.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 83.43 | 84.24 | 83.31 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 80.75 | 82.51 | 82.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 80.50 | 82.11 | 82.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 82.18 | 81.93 | 82.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 10:15:00 | 82.18 | 81.93 | 82.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 82.18 | 81.93 | 82.30 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 91.70 | 83.92 | 83.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 11:15:00 | 92.95 | 86.96 | 84.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 13:15:00 | 86.98 | 86.98 | 85.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 15:15:00 | 89.80 | 90.66 | 89.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 89.80 | 90.66 | 89.76 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 11:15:00 | 89.03 | 89.74 | 89.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 14:15:00 | 87.25 | 89.06 | 89.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 11:15:00 | 89.03 | 88.86 | 89.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 11:15:00 | 89.03 | 88.86 | 89.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 89.03 | 88.86 | 89.23 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 95.55 | 89.87 | 89.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 14:15:00 | 97.18 | 95.73 | 93.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 10:15:00 | 95.60 | 96.29 | 94.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 12:15:00 | 99.33 | 101.05 | 100.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 99.33 | 101.05 | 100.36 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 13:15:00 | 99.60 | 100.14 | 100.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 15:15:00 | 98.73 | 99.70 | 99.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 97.95 | 97.95 | 98.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 97.95 | 97.95 | 98.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 97.95 | 97.95 | 98.69 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 12:15:00 | 94.88 | 94.40 | 94.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 97.35 | 95.15 | 94.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 09:15:00 | 96.40 | 96.85 | 95.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 96.40 | 96.85 | 95.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 96.40 | 96.85 | 95.94 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 94.75 | 96.57 | 96.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 94.13 | 96.09 | 96.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 12:15:00 | 95.80 | 94.88 | 95.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 12:15:00 | 95.80 | 94.88 | 95.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 95.80 | 94.88 | 95.22 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 96.03 | 94.78 | 94.70 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 94.33 | 94.69 | 94.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 10:15:00 | 93.10 | 94.22 | 94.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 11:15:00 | 92.93 | 92.91 | 93.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 11:15:00 | 92.93 | 92.91 | 93.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 92.93 | 92.91 | 93.54 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 96.60 | 93.97 | 93.76 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 91.35 | 93.70 | 93.87 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 10:15:00 | 94.75 | 94.06 | 94.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 10:15:00 | 98.25 | 95.10 | 94.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 108.80 | 109.18 | 106.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 11:15:00 | 107.50 | 108.72 | 107.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 107.50 | 108.72 | 107.25 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 14:15:00 | 133.50 | 134.61 | 134.63 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 15:15:00 | 135.25 | 134.74 | 134.68 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 10:15:00 | 134.35 | 134.61 | 134.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 12:15:00 | 132.90 | 133.67 | 134.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 14:15:00 | 134.00 | 133.59 | 133.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 14:15:00 | 134.00 | 133.59 | 133.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 134.00 | 133.59 | 133.93 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 136.07 | 134.49 | 134.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 138.90 | 135.48 | 134.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 135.32 | 136.08 | 135.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 135.32 | 136.08 | 135.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 135.32 | 136.08 | 135.33 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 132.75 | 134.68 | 134.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 130.43 | 133.36 | 134.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 132.18 | 131.50 | 132.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 12:15:00 | 131.95 | 131.59 | 132.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 131.95 | 131.59 | 132.60 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 135.55 | 132.85 | 132.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 137.15 | 134.28 | 133.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 135.73 | 135.86 | 134.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 14:15:00 | 135.73 | 135.86 | 134.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 135.73 | 135.86 | 134.95 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 11:15:00 | 166.98 | 173.37 | 173.60 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 179.50 | 174.54 | 173.90 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 13:15:00 | 169.58 | 174.66 | 174.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 168.73 | 173.47 | 174.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 168.05 | 164.77 | 167.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 168.05 | 164.77 | 167.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 168.05 | 164.77 | 167.86 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 159.03 | 152.27 | 152.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 163.45 | 158.50 | 155.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 15:15:00 | 169.50 | 169.51 | 165.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 166.40 | 168.88 | 165.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 166.40 | 168.88 | 165.71 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 09:15:00 | 158.28 | 163.45 | 164.13 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 14:15:00 | 168.03 | 164.05 | 163.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 11:15:00 | 170.93 | 166.04 | 165.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 15:15:00 | 171.00 | 171.19 | 169.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 178.50 | 178.66 | 175.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 178.50 | 178.66 | 175.15 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 171.08 | 175.13 | 175.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 170.00 | 172.46 | 173.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 172.50 | 169.65 | 171.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 172.50 | 169.65 | 171.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 172.50 | 169.65 | 171.48 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 177.00 | 172.15 | 171.95 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 172.00 | 173.64 | 173.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 169.73 | 172.84 | 173.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 173.00 | 170.37 | 171.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 173.00 | 170.37 | 171.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 173.00 | 170.37 | 171.62 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 12:15:00 | 165.00 | 162.76 | 162.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 14:15:00 | 168.10 | 164.26 | 163.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 09:15:00 | 163.50 | 164.72 | 163.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 163.50 | 164.72 | 163.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 163.50 | 164.72 | 163.78 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 10:15:00 | 161.50 | 163.28 | 163.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 11:15:00 | 160.00 | 162.62 | 163.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 14:15:00 | 163.40 | 162.28 | 162.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 14:15:00 | 163.40 | 162.28 | 162.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 163.40 | 162.28 | 162.82 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 162.93 | 159.47 | 159.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 170.00 | 163.21 | 161.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 10:15:00 | 202.50 | 206.55 | 199.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 213.00 | 211.31 | 205.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 213.00 | 211.31 | 205.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 256.50 | 256.17 | 247.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-15 09:15:00 | 282.15 | 272.88 | 262.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 10:15:00 | 330.55 | 343.53 | 345.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 09:15:00 | 314.02 | 330.65 | 337.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 09:15:00 | 329.73 | 320.65 | 327.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 329.73 | 320.65 | 327.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 329.73 | 320.65 | 327.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:30:00 | 329.73 | 320.65 | 327.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 329.73 | 322.46 | 327.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:30:00 | 329.73 | 322.46 | 327.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 320.50 | 317.44 | 321.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:30:00 | 324.40 | 317.44 | 321.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 322.95 | 318.54 | 321.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:00:00 | 322.95 | 318.54 | 321.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 324.25 | 319.68 | 321.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:45:00 | 324.90 | 319.68 | 321.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 325.70 | 320.89 | 322.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:00:00 | 325.70 | 320.89 | 322.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 14:15:00 | 329.23 | 323.37 | 323.03 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 316.00 | 322.01 | 322.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 303.23 | 316.85 | 319.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 302.48 | 294.10 | 300.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 302.48 | 294.10 | 300.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 302.48 | 294.10 | 300.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 300.40 | 294.10 | 300.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 302.48 | 295.78 | 300.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:30:00 | 302.48 | 295.78 | 300.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 302.48 | 297.12 | 301.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:30:00 | 302.48 | 297.12 | 301.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 302.48 | 299.43 | 301.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:45:00 | 301.50 | 299.43 | 301.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 302.48 | 300.04 | 301.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:15:00 | 298.48 | 300.04 | 301.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 292.48 | 291.52 | 295.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:15:00 | 279.45 | 289.48 | 292.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 302.00 | 290.66 | 291.52 | SL hit (close>static) qty=1.00 sl=298.10 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 299.45 | 292.42 | 292.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 308.48 | 296.16 | 294.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 307.70 | 311.28 | 306.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 307.70 | 311.28 | 306.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 310.00 | 310.89 | 307.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 11:45:00 | 314.98 | 312.00 | 308.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 313.00 | 312.13 | 309.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:45:00 | 313.25 | 312.20 | 310.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 314.98 | 312.20 | 310.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 312.00 | 312.66 | 310.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 312.00 | 312.66 | 310.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 310.13 | 312.16 | 310.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 314.00 | 312.16 | 310.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 312.50 | 312.23 | 311.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 310.50 | 311.77 | 310.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 312.30 | 311.88 | 311.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 312.00 | 311.88 | 311.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 310.50 | 311.60 | 311.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-21 13:15:00 | 309.00 | 310.72 | 310.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 309.00 | 310.72 | 310.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 303.50 | 308.57 | 309.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 309.33 | 298.96 | 301.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 309.33 | 298.96 | 301.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 309.33 | 298.96 | 301.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 309.33 | 298.96 | 301.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 12:15:00 | 309.33 | 304.02 | 303.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 10:15:00 | 321.48 | 309.54 | 306.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 318.10 | 320.44 | 315.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 11:45:00 | 317.52 | 320.44 | 315.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 322.00 | 322.04 | 318.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 325.40 | 322.12 | 319.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-03 09:15:00 | 357.94 | 349.80 | 340.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 331.68 | 348.72 | 349.80 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 360.00 | 351.18 | 350.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 366.58 | 355.51 | 352.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 12:15:00 | 361.00 | 362.02 | 358.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 13:00:00 | 361.00 | 362.02 | 358.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 362.50 | 362.00 | 358.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 357.00 | 362.00 | 358.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 353.50 | 360.30 | 358.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 353.50 | 360.30 | 358.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 365.50 | 365.50 | 362.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:30:00 | 367.00 | 365.50 | 362.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 397.65 | 402.39 | 396.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 397.88 | 402.39 | 396.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 401.20 | 402.16 | 397.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:30:00 | 397.38 | 402.16 | 397.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 400.00 | 402.18 | 397.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:45:00 | 400.00 | 402.18 | 397.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 399.00 | 401.54 | 398.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:30:00 | 400.00 | 401.54 | 398.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 398.25 | 400.60 | 398.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 388.00 | 400.60 | 398.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 386.98 | 397.87 | 397.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 386.98 | 397.87 | 397.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 385.50 | 395.40 | 396.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 11:15:00 | 383.00 | 391.52 | 393.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 12:15:00 | 386.25 | 384.99 | 388.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 13:00:00 | 386.25 | 384.99 | 388.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 388.50 | 386.09 | 388.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:45:00 | 387.50 | 386.09 | 388.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 386.50 | 386.17 | 388.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 390.00 | 387.34 | 388.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 388.55 | 387.58 | 388.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 390.50 | 387.58 | 388.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 387.50 | 387.57 | 388.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 387.50 | 387.57 | 388.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 385.50 | 387.15 | 388.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 376.98 | 385.76 | 387.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 14:15:00 | 358.13 | 366.36 | 369.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 15:15:00 | 364.05 | 362.24 | 365.32 | SL hit (close>ema200) qty=0.50 sl=362.24 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 379.48 | 368.45 | 367.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 388.00 | 378.29 | 373.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 14:15:00 | 394.25 | 398.93 | 391.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 14:15:00 | 394.25 | 398.93 | 391.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 394.25 | 398.93 | 391.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 394.25 | 398.93 | 391.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 399.00 | 398.94 | 392.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 400.03 | 398.94 | 392.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 400.53 | 398.53 | 394.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 400.00 | 398.77 | 394.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 390.03 | 393.76 | 394.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 390.03 | 393.76 | 394.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 375.00 | 383.05 | 386.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 13:15:00 | 377.00 | 376.78 | 379.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 15:00:00 | 371.80 | 375.78 | 379.16 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 371.50 | 375.08 | 378.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 385.50 | 377.63 | 378.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 385.50 | 377.63 | 378.64 | SL hit (close>ema400) qty=1.00 sl=378.64 alert=retest1 |

### Cycle 79 — BUY (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 14:15:00 | 381.75 | 379.55 | 379.41 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 374.00 | 378.83 | 379.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 10:15:00 | 369.25 | 376.91 | 378.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 14:15:00 | 375.50 | 375.33 | 376.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:30:00 | 375.25 | 375.33 | 376.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 375.50 | 375.37 | 376.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 382.50 | 375.37 | 376.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 378.00 | 375.89 | 376.89 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 378.50 | 377.45 | 377.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 13:15:00 | 380.48 | 378.05 | 377.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 372.50 | 378.06 | 377.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 372.50 | 378.06 | 377.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 372.50 | 378.06 | 377.98 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 374.50 | 377.35 | 377.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 357.50 | 371.58 | 374.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 365.50 | 362.66 | 367.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 365.50 | 362.66 | 367.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 365.50 | 362.66 | 367.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 366.70 | 362.66 | 367.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 366.00 | 363.33 | 367.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:00:00 | 362.00 | 363.06 | 366.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 357.23 | 362.33 | 365.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 360.20 | 357.79 | 360.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 375.50 | 362.56 | 362.59 | SL hit (close>static) qty=1.00 sl=367.50 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 377.50 | 365.55 | 363.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 378.50 | 368.14 | 365.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 380.50 | 381.57 | 376.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 14:30:00 | 383.00 | 381.57 | 376.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 373.00 | 379.36 | 376.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 373.00 | 379.36 | 376.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 382.00 | 379.89 | 376.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 373.00 | 379.89 | 376.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 377.08 | 379.26 | 377.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:00:00 | 377.08 | 379.26 | 377.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 373.00 | 378.01 | 377.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:45:00 | 372.50 | 378.01 | 377.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 372.50 | 376.91 | 376.86 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 373.75 | 376.28 | 376.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 365.00 | 371.90 | 373.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 14:15:00 | 369.00 | 368.65 | 371.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-01 15:00:00 | 369.00 | 368.65 | 371.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 364.20 | 367.57 | 370.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:30:00 | 364.50 | 367.57 | 370.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 357.40 | 351.92 | 357.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 357.05 | 351.92 | 357.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 358.43 | 353.22 | 357.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 352.50 | 354.92 | 357.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 355.00 | 352.98 | 355.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:15:00 | 354.50 | 353.68 | 355.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:30:00 | 355.18 | 354.70 | 355.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 354.50 | 354.66 | 355.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:45:00 | 356.33 | 354.66 | 355.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 356.00 | 354.93 | 355.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 354.00 | 354.93 | 355.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 350.00 | 353.94 | 355.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:30:00 | 344.50 | 352.05 | 353.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 346.50 | 350.77 | 352.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 345.65 | 350.77 | 352.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:15:00 | 345.00 | 350.12 | 352.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 334.88 | 345.75 | 349.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 337.25 | 345.75 | 349.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 336.77 | 345.75 | 349.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 337.42 | 345.75 | 349.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 344.98 | 344.28 | 347.40 | SL hit (close>ema200) qty=0.50 sl=344.28 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 346.50 | 343.64 | 343.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 353.00 | 346.32 | 344.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 13:15:00 | 347.50 | 348.16 | 346.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 14:00:00 | 347.50 | 348.16 | 346.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 346.00 | 347.73 | 346.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:30:00 | 347.03 | 347.73 | 346.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 348.50 | 347.88 | 346.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 345.50 | 347.88 | 346.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 341.75 | 346.65 | 346.04 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 342.50 | 346.15 | 346.34 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 10:15:00 | 354.93 | 346.05 | 345.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 372.50 | 356.12 | 351.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 363.00 | 367.25 | 360.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 363.00 | 367.25 | 360.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 363.00 | 367.25 | 360.70 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 359.13 | 362.28 | 362.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 357.50 | 361.20 | 361.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 14:15:00 | 361.50 | 361.26 | 361.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 14:15:00 | 361.50 | 361.26 | 361.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 361.50 | 361.26 | 361.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 11:15:00 | 355.55 | 360.42 | 361.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:15:00 | 355.03 | 359.93 | 360.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 372.60 | 360.50 | 359.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 372.60 | 360.50 | 359.77 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 357.50 | 360.70 | 360.78 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 365.50 | 361.04 | 360.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 15:15:00 | 367.50 | 362.33 | 361.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 15:15:00 | 361.50 | 362.79 | 362.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 15:15:00 | 361.50 | 362.79 | 362.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 361.50 | 362.79 | 362.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 360.50 | 362.79 | 362.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 362.45 | 362.72 | 362.24 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 358.63 | 361.87 | 361.93 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 362.50 | 362.00 | 361.98 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 361.00 | 361.80 | 361.89 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 366.98 | 362.83 | 362.35 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 359.48 | 362.21 | 362.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 356.90 | 358.89 | 360.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 350.13 | 347.39 | 352.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 350.13 | 347.39 | 352.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 355.00 | 348.91 | 352.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 348.75 | 348.91 | 352.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:15:00 | 331.31 | 338.96 | 343.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 336.50 | 335.25 | 339.14 | SL hit (close>ema200) qty=0.50 sl=335.25 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 320.00 | 315.15 | 314.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 321.50 | 317.46 | 316.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 326.50 | 327.76 | 324.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 326.50 | 327.76 | 324.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 326.50 | 327.76 | 324.78 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 318.48 | 324.07 | 324.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 11:15:00 | 314.50 | 321.90 | 323.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 11:15:00 | 315.18 | 313.19 | 316.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 11:15:00 | 315.18 | 313.19 | 316.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 315.18 | 313.19 | 316.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 315.18 | 313.19 | 316.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 321.98 | 314.95 | 316.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 322.50 | 314.95 | 316.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 313.55 | 314.67 | 316.31 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 323.33 | 317.46 | 317.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 325.25 | 319.02 | 317.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 15:15:00 | 332.50 | 332.69 | 327.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:15:00 | 329.50 | 332.69 | 327.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 333.50 | 332.85 | 327.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 328.48 | 332.85 | 327.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 331.03 | 332.49 | 328.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 327.50 | 332.49 | 328.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 331.00 | 332.66 | 329.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 331.00 | 332.66 | 329.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 335.00 | 333.13 | 329.82 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 321.00 | 328.10 | 328.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 317.65 | 322.66 | 325.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 11:15:00 | 323.27 | 322.35 | 324.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 11:15:00 | 323.27 | 322.35 | 324.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 323.27 | 322.35 | 324.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 323.27 | 322.35 | 324.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 328.03 | 323.48 | 324.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:45:00 | 329.35 | 323.48 | 324.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 337.05 | 326.20 | 326.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 353.90 | 334.86 | 330.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 15:15:00 | 432.50 | 435.01 | 423.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:15:00 | 430.00 | 435.01 | 423.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 420.43 | 432.09 | 423.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 420.43 | 432.09 | 423.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 431.25 | 431.93 | 424.04 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 416.45 | 422.68 | 422.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 410.00 | 416.04 | 418.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 13:15:00 | 411.25 | 405.73 | 410.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 13:15:00 | 411.25 | 405.73 | 410.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 411.25 | 405.73 | 410.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:00:00 | 411.25 | 405.73 | 410.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 415.00 | 407.58 | 410.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 415.00 | 407.58 | 410.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 425.00 | 411.07 | 411.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 427.50 | 411.07 | 411.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 09:15:00 | 423.35 | 413.52 | 412.90 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 400.00 | 415.09 | 416.95 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 417.50 | 415.46 | 415.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 439.40 | 422.71 | 419.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 17:15:00 | 454.00 | 457.14 | 447.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 454.00 | 457.14 | 447.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 446.98 | 455.17 | 448.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 443.00 | 455.17 | 448.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 443.00 | 452.73 | 447.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 443.00 | 452.73 | 447.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 456.38 | 453.46 | 448.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:45:00 | 457.40 | 456.57 | 450.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-07 12:15:00 | 503.14 | 494.48 | 484.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 482.48 | 500.36 | 500.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 468.45 | 489.15 | 495.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 15:15:00 | 438.95 | 437.50 | 450.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 09:15:00 | 435.00 | 437.50 | 450.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 451.95 | 440.55 | 449.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:30:00 | 445.53 | 440.55 | 449.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 452.30 | 442.90 | 450.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:30:00 | 452.30 | 442.90 | 450.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 474.90 | 453.74 | 452.96 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 456.00 | 458.17 | 458.42 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 463.00 | 458.10 | 457.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 475.33 | 461.55 | 459.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 484.28 | 487.56 | 480.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 484.28 | 487.56 | 480.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 484.28 | 487.56 | 480.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:30:00 | 494.50 | 487.81 | 484.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:00:00 | 497.00 | 487.81 | 484.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:45:00 | 494.50 | 493.64 | 489.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 14:15:00 | 489.80 | 490.27 | 490.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 489.80 | 490.27 | 490.29 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 505.00 | 493.17 | 491.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 512.50 | 505.23 | 499.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 13:15:00 | 504.58 | 508.09 | 502.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 13:15:00 | 504.58 | 508.09 | 502.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 504.58 | 508.09 | 502.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:00:00 | 504.58 | 508.09 | 502.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 514.00 | 509.27 | 503.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 515.00 | 509.27 | 503.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:00:00 | 515.25 | 510.84 | 506.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 13:45:00 | 519.25 | 519.46 | 517.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-12 09:15:00 | 566.50 | 539.42 | 530.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 574.25 | 574.63 | 574.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 564.48 | 572.60 | 573.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 541.98 | 541.50 | 551.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 15:00:00 | 541.98 | 541.50 | 551.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 539.03 | 530.78 | 535.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 554.50 | 530.78 | 535.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 555.00 | 535.63 | 537.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 561.85 | 535.63 | 537.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 543.50 | 537.20 | 537.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:15:00 | 550.00 | 537.20 | 537.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 539.90 | 537.74 | 537.80 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 549.48 | 540.09 | 538.87 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 530.00 | 538.21 | 538.99 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 555.05 | 540.16 | 539.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 14:15:00 | 573.75 | 551.59 | 545.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 617.70 | 620.31 | 607.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 617.70 | 620.31 | 607.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 611.45 | 618.49 | 608.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 612.45 | 618.49 | 608.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 604.63 | 615.72 | 608.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 604.63 | 615.72 | 608.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 600.00 | 612.57 | 607.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:30:00 | 597.42 | 612.57 | 607.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 598.00 | 609.66 | 606.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:00:00 | 598.00 | 609.66 | 606.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 614.98 | 607.01 | 605.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:00:00 | 647.50 | 619.90 | 614.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 588.10 | 614.08 | 613.14 | SL hit (close<static) qty=1.00 sl=600.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 592.50 | 609.77 | 611.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 559.92 | 590.86 | 600.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 509.68 | 501.91 | 516.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-16 09:30:00 | 512.50 | 501.91 | 516.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 510.53 | 509.05 | 513.64 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 534.00 | 516.84 | 514.75 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 515.00 | 518.97 | 519.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 496.30 | 514.49 | 517.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 509.50 | 502.79 | 508.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 509.50 | 502.79 | 508.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 509.50 | 502.79 | 508.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:30:00 | 511.50 | 502.79 | 508.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 499.75 | 502.19 | 507.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 504.00 | 502.19 | 507.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 503.50 | 501.72 | 506.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:00:00 | 503.50 | 501.72 | 506.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 477.98 | 496.31 | 502.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:45:00 | 473.33 | 485.91 | 494.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 449.66 | 450.50 | 466.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 449.50 | 443.59 | 459.12 | SL hit (close>ema200) qty=0.50 sl=443.59 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 459.58 | 452.74 | 452.42 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 449.30 | 452.06 | 452.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 426.53 | 446.95 | 449.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 15:15:00 | 449.50 | 446.38 | 448.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 15:15:00 | 449.50 | 446.38 | 448.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 449.50 | 446.38 | 448.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 445.95 | 446.38 | 448.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 442.65 | 445.63 | 448.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 14:00:00 | 435.00 | 442.59 | 446.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 436.03 | 443.80 | 446.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 09:45:00 | 436.00 | 443.12 | 445.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 10:15:00 | 433.80 | 443.12 | 445.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 413.25 | 423.98 | 433.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 414.23 | 423.98 | 433.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 414.20 | 423.98 | 433.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 412.11 | 423.98 | 433.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-04 09:15:00 | 391.50 | 404.07 | 416.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 427.80 | 408.99 | 407.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 433.70 | 424.42 | 417.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 429.68 | 434.51 | 427.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 429.68 | 434.51 | 427.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 429.68 | 434.51 | 427.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 428.25 | 434.51 | 427.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 427.98 | 432.85 | 427.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 427.98 | 432.85 | 427.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 424.50 | 431.18 | 427.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:00:00 | 424.50 | 431.18 | 427.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 421.25 | 429.19 | 426.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:30:00 | 433.98 | 434.35 | 429.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 408.23 | 427.54 | 429.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 408.23 | 427.54 | 429.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 403.75 | 418.33 | 421.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 385.80 | 373.88 | 383.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 385.80 | 373.88 | 383.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 385.80 | 373.88 | 383.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 385.80 | 373.88 | 383.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 385.40 | 376.18 | 383.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 385.80 | 376.18 | 383.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 385.80 | 378.10 | 383.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 385.80 | 378.10 | 383.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 405.05 | 387.13 | 386.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 09:15:00 | 417.40 | 403.76 | 396.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 10:15:00 | 437.95 | 440.27 | 428.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:45:00 | 436.60 | 440.27 | 428.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 430.00 | 436.31 | 429.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:30:00 | 428.85 | 436.31 | 429.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 426.70 | 434.38 | 429.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 426.70 | 434.38 | 429.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 423.80 | 432.27 | 428.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 433.00 | 432.27 | 428.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:45:00 | 432.65 | 431.96 | 429.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 10:45:00 | 428.45 | 430.47 | 428.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 12:15:00 | 410.50 | 424.80 | 426.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 410.50 | 424.80 | 426.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 394.95 | 414.24 | 420.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 390.35 | 374.64 | 386.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 390.35 | 374.64 | 386.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 390.35 | 374.64 | 386.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 387.95 | 374.64 | 386.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 382.90 | 376.30 | 385.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 379.45 | 377.30 | 385.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:30:00 | 378.50 | 378.99 | 384.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 409.10 | 388.26 | 387.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 409.10 | 388.26 | 387.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 13:15:00 | 415.60 | 406.53 | 400.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 410.90 | 417.34 | 411.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 410.90 | 417.34 | 411.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 410.90 | 417.34 | 411.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 410.90 | 417.34 | 411.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 400.55 | 413.98 | 410.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 400.55 | 413.98 | 410.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 398.70 | 410.93 | 409.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:45:00 | 398.95 | 410.93 | 409.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 396.55 | 408.05 | 408.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 394.00 | 405.24 | 407.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 393.50 | 392.61 | 397.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 390.20 | 392.61 | 397.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 387.60 | 391.61 | 396.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 381.05 | 391.35 | 396.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 371.10 | 386.19 | 391.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 388.80 | 386.72 | 386.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 388.80 | 386.72 | 386.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 408.00 | 391.86 | 389.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 12:15:00 | 514.00 | 514.55 | 503.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 13:00:00 | 514.00 | 514.55 | 503.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 496.70 | 510.98 | 503.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:00:00 | 496.70 | 510.98 | 503.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 488.45 | 506.47 | 501.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:30:00 | 488.45 | 506.47 | 501.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 497.05 | 501.91 | 500.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 500.00 | 501.91 | 500.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 496.25 | 500.78 | 500.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 496.25 | 500.78 | 500.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 520.70 | 527.70 | 519.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 519.35 | 527.70 | 519.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 510.05 | 524.17 | 518.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:45:00 | 512.65 | 524.17 | 518.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 509.40 | 521.22 | 517.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 509.40 | 521.22 | 517.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 509.40 | 514.24 | 514.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 502.80 | 511.95 | 513.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 511.85 | 510.92 | 512.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 15:00:00 | 511.85 | 510.92 | 512.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 510.05 | 510.74 | 512.28 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 524.55 | 513.50 | 513.39 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 494.40 | 512.99 | 514.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 470.95 | 495.32 | 503.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 485.10 | 478.89 | 489.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 485.10 | 478.89 | 489.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 485.10 | 478.89 | 489.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 494.00 | 478.89 | 489.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 481.65 | 479.44 | 488.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:30:00 | 484.25 | 479.44 | 488.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 490.00 | 481.55 | 488.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 490.00 | 481.55 | 488.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 487.90 | 482.82 | 488.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 491.35 | 482.82 | 488.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 494.20 | 485.10 | 489.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:45:00 | 494.20 | 485.10 | 489.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 494.45 | 486.97 | 489.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:30:00 | 493.50 | 486.97 | 489.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 518.55 | 494.48 | 492.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 544.60 | 518.35 | 507.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 551.75 | 559.67 | 546.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 551.75 | 559.67 | 546.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 551.75 | 559.67 | 546.08 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 10:15:00 | 531.00 | 543.99 | 544.37 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 548.00 | 542.73 | 542.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 573.50 | 550.30 | 545.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 557.30 | 565.55 | 557.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 557.30 | 565.55 | 557.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 557.30 | 565.55 | 557.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 557.30 | 565.55 | 557.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 563.80 | 565.20 | 558.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 557.50 | 565.20 | 558.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 555.60 | 563.28 | 558.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:45:00 | 554.50 | 563.28 | 558.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 557.05 | 562.04 | 558.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 555.60 | 562.04 | 558.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 556.10 | 560.85 | 557.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 556.10 | 560.85 | 557.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 559.55 | 560.59 | 558.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 555.45 | 560.59 | 558.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 563.80 | 561.23 | 558.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 547.95 | 561.23 | 558.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 551.00 | 559.18 | 557.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:30:00 | 546.90 | 559.18 | 557.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 10:15:00 | 546.35 | 556.62 | 556.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 11:15:00 | 540.95 | 553.48 | 555.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 527.45 | 519.47 | 530.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 527.45 | 519.47 | 530.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 527.45 | 519.47 | 530.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 527.45 | 519.47 | 530.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 499.80 | 496.84 | 503.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:15:00 | 499.80 | 496.84 | 503.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 497.95 | 497.07 | 503.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:30:00 | 496.70 | 496.56 | 502.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 495.15 | 497.11 | 501.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 14:15:00 | 508.80 | 500.95 | 500.97 | SL hit (close>static) qty=1.00 sl=505.95 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 508.85 | 502.53 | 501.68 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 492.80 | 500.59 | 500.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 490.50 | 496.19 | 498.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 491.85 | 489.27 | 493.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 491.85 | 489.27 | 493.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 497.20 | 490.86 | 494.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 497.20 | 490.86 | 494.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 492.05 | 491.10 | 493.84 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 502.00 | 494.93 | 494.68 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 483.50 | 493.06 | 494.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 480.45 | 490.54 | 492.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 481.10 | 479.93 | 484.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 502.35 | 479.93 | 484.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 502.35 | 484.41 | 486.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 502.35 | 484.41 | 486.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 502.35 | 488.00 | 487.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 519.75 | 502.07 | 495.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 510.00 | 511.23 | 506.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:00:00 | 510.00 | 511.23 | 506.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 496.50 | 507.66 | 506.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 496.80 | 507.66 | 506.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 496.00 | 505.33 | 505.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 496.00 | 505.33 | 505.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 11:15:00 | 498.00 | 503.86 | 504.56 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 505.30 | 503.64 | 503.61 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 499.00 | 502.71 | 503.19 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 522.05 | 506.13 | 504.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 535.90 | 523.10 | 514.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 520.20 | 525.38 | 518.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 520.20 | 525.38 | 518.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 512.55 | 522.82 | 518.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 512.55 | 522.82 | 518.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 514.15 | 521.08 | 517.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 508.20 | 521.08 | 517.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 508.95 | 517.89 | 516.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 508.95 | 517.89 | 516.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 499.90 | 514.29 | 515.36 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 514.80 | 510.88 | 510.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 521.00 | 514.21 | 512.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 513.70 | 517.86 | 515.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 513.70 | 517.86 | 515.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 513.70 | 517.86 | 515.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 525.75 | 518.64 | 516.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 522.90 | 518.64 | 516.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 511.35 | 517.28 | 517.12 | SL hit (close<static) qty=1.00 sl=512.60 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 513.00 | 516.42 | 516.74 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 524.85 | 518.03 | 517.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 532.05 | 520.83 | 518.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 523.95 | 523.98 | 521.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:45:00 | 529.85 | 523.98 | 521.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 523.70 | 523.92 | 521.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 522.95 | 523.92 | 521.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 523.10 | 523.76 | 521.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 523.10 | 523.76 | 521.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 522.65 | 523.54 | 521.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 522.65 | 523.54 | 521.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 522.90 | 523.41 | 521.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 521.55 | 523.41 | 521.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 520.15 | 522.76 | 521.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 518.95 | 522.76 | 521.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 522.40 | 522.69 | 521.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 528.20 | 522.69 | 521.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 534.60 | 525.07 | 522.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:30:00 | 535.00 | 527.57 | 524.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 536.60 | 529.18 | 526.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 536.10 | 527.93 | 527.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 539.10 | 530.14 | 528.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 531.50 | 533.16 | 531.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 531.50 | 533.16 | 531.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 531.40 | 532.81 | 531.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 530.40 | 532.81 | 531.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 531.00 | 532.45 | 531.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 524.75 | 532.45 | 531.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 524.05 | 530.77 | 530.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 524.05 | 530.77 | 530.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 518.55 | 525.99 | 528.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 13:15:00 | 518.75 | 518.55 | 522.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:00:00 | 518.75 | 518.55 | 522.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 521.85 | 518.95 | 521.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 521.25 | 518.95 | 521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 519.90 | 519.14 | 521.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:15:00 | 517.95 | 519.14 | 521.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:45:00 | 518.20 | 518.05 | 519.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.05 | 505.69 | 510.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.29 | 505.69 | 510.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 494.95 | 493.36 | 498.29 | SL hit (close>ema200) qty=0.50 sl=493.36 alert=retest2 |

### Cycle 149 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 496.90 | 486.47 | 485.98 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 12:15:00 | 487.95 | 488.67 | 488.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 484.80 | 487.81 | 488.26 | Break + close below crossover candle low |

### Cycle 151 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 493.05 | 488.77 | 488.61 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 486.90 | 488.41 | 488.48 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 488.65 | 488.46 | 488.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 492.40 | 489.25 | 488.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 489.50 | 490.92 | 489.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 489.50 | 490.92 | 489.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 489.50 | 490.92 | 489.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 489.60 | 490.92 | 489.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 490.00 | 490.73 | 489.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 499.15 | 490.73 | 489.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 15:15:00 | 488.80 | 490.54 | 490.39 | SL hit (close<static) qty=1.00 sl=489.10 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 469.20 | 486.27 | 488.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 465.50 | 469.67 | 475.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 469.10 | 468.85 | 474.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 469.10 | 468.85 | 474.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 473.00 | 470.08 | 474.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 469.05 | 470.62 | 472.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 468.30 | 469.66 | 470.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 469.20 | 469.66 | 470.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:45:00 | 469.10 | 469.46 | 470.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 482.15 | 472.00 | 471.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 482.15 | 472.00 | 471.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 14:15:00 | 487.90 | 478.22 | 474.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 511.50 | 512.09 | 502.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 12:15:00 | 514.30 | 512.09 | 502.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 501.05 | 509.76 | 505.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 501.05 | 509.76 | 505.28 | SL hit (close<ema400) qty=1.00 sl=505.28 alert=retest1 |

### Cycle 156 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 501.00 | 503.35 | 503.49 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 503.95 | 503.64 | 503.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 508.60 | 504.76 | 504.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 506.25 | 506.78 | 505.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 506.25 | 506.78 | 505.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 506.25 | 506.78 | 505.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 505.10 | 506.78 | 505.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 510.75 | 507.57 | 505.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 513.00 | 509.60 | 507.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 517.50 | 510.33 | 508.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 511.55 | 513.32 | 513.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 511.55 | 513.32 | 513.34 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 514.10 | 513.48 | 513.41 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 512.00 | 513.30 | 513.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 510.25 | 512.64 | 513.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 491.65 | 489.36 | 495.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 14:45:00 | 493.35 | 489.36 | 495.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 487.75 | 485.85 | 489.63 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 510.40 | 491.17 | 490.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 513.35 | 508.15 | 503.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 509.40 | 510.00 | 506.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:30:00 | 519.50 | 512.10 | 507.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 516.75 | 516.61 | 511.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:45:00 | 513.45 | 516.61 | 511.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 518.70 | 517.03 | 511.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 513.85 | 517.03 | 511.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 519.30 | 517.48 | 512.65 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 12:15:00 | 545.48 | 523.98 | 516.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:45:00 | 532.25 | 523.98 | 516.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-05 09:15:00 | 571.45 | 542.38 | 528.95 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 162 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 505.45 | 530.65 | 531.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 499.10 | 512.81 | 520.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 505.15 | 501.94 | 511.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 505.15 | 501.94 | 511.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 505.15 | 501.94 | 511.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 504.20 | 501.94 | 511.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 503.20 | 503.13 | 510.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 499.75 | 502.36 | 509.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 529.70 | 504.45 | 506.16 | SL hit (close>static) qty=1.00 sl=511.80 alert=retest2 |

### Cycle 163 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 526.10 | 508.78 | 507.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 534.00 | 513.82 | 510.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 516.60 | 519.68 | 514.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 516.60 | 519.68 | 514.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 516.60 | 519.68 | 514.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 516.60 | 519.68 | 514.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 507.45 | 517.24 | 514.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 507.45 | 517.24 | 514.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 507.15 | 515.22 | 513.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 505.35 | 515.22 | 513.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 507.30 | 513.43 | 512.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 507.30 | 513.43 | 512.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 506.25 | 512.00 | 512.36 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 513.90 | 512.72 | 512.60 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 507.90 | 512.06 | 512.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 505.85 | 508.37 | 510.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 509.00 | 508.50 | 510.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 509.00 | 508.50 | 510.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 509.00 | 508.50 | 510.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 505.10 | 507.76 | 509.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 510.00 | 503.16 | 502.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 510.00 | 503.16 | 502.28 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 502.60 | 503.12 | 503.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 500.00 | 502.12 | 502.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 499.35 | 495.50 | 497.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 499.35 | 495.50 | 497.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 499.35 | 495.50 | 497.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 499.35 | 495.50 | 497.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 496.60 | 495.72 | 497.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 494.85 | 495.72 | 497.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 494.70 | 493.81 | 495.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 501.70 | 495.38 | 496.30 | SL hit (close>static) qty=1.00 sl=501.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 504.15 | 497.14 | 497.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 507.30 | 502.44 | 500.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 500.05 | 504.79 | 503.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 500.05 | 504.79 | 503.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 500.05 | 504.79 | 503.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 500.05 | 504.79 | 503.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 500.35 | 503.90 | 503.05 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 500.90 | 502.33 | 502.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 495.15 | 500.49 | 501.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 501.90 | 499.70 | 500.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 501.90 | 499.70 | 500.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 501.90 | 499.70 | 500.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 501.90 | 499.70 | 500.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 504.00 | 500.56 | 501.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 504.00 | 500.56 | 501.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 502.60 | 500.97 | 501.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 499.20 | 500.97 | 501.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 505.85 | 497.02 | 497.20 | SL hit (close>static) qty=1.00 sl=504.15 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 509.10 | 499.43 | 498.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 519.20 | 504.93 | 501.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 508.50 | 510.73 | 505.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 508.50 | 510.73 | 505.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 505.50 | 509.68 | 505.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 505.10 | 509.68 | 505.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 501.75 | 508.10 | 505.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 501.90 | 508.10 | 505.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 501.05 | 506.69 | 504.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 501.00 | 506.69 | 504.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 505.00 | 505.64 | 504.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 501.90 | 505.64 | 504.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 499.55 | 504.42 | 504.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 498.35 | 504.42 | 504.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 498.95 | 503.33 | 503.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 498.05 | 500.72 | 502.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 502.05 | 500.70 | 501.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 502.05 | 500.70 | 501.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 502.05 | 500.70 | 501.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 503.30 | 500.70 | 501.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 506.00 | 501.76 | 502.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 506.00 | 501.76 | 502.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 505.45 | 502.50 | 502.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 507.35 | 502.50 | 502.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 503.85 | 502.77 | 502.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 517.60 | 506.67 | 504.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 524.85 | 525.72 | 520.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 524.85 | 525.72 | 520.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 520.80 | 524.04 | 521.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 517.35 | 524.04 | 521.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 520.80 | 523.39 | 520.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 519.55 | 523.39 | 520.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 520.30 | 522.77 | 520.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 526.50 | 522.38 | 520.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 523.50 | 531.52 | 531.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 523.50 | 531.52 | 531.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 517.35 | 525.38 | 528.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 490.70 | 487.84 | 493.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:30:00 | 492.40 | 487.84 | 493.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 489.35 | 488.14 | 493.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 490.30 | 488.14 | 493.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 492.15 | 489.49 | 491.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 495.00 | 489.49 | 491.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 492.95 | 490.18 | 491.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 496.80 | 490.18 | 491.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 493.80 | 490.91 | 492.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 493.80 | 490.91 | 492.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 493.15 | 491.36 | 492.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 494.75 | 491.36 | 492.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 495.20 | 492.78 | 492.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 496.75 | 493.57 | 493.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 494.95 | 495.05 | 494.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:00:00 | 494.95 | 495.05 | 494.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 494.25 | 494.89 | 494.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 494.25 | 494.89 | 494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 494.00 | 494.71 | 494.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 494.70 | 494.71 | 494.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 494.15 | 494.60 | 494.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:15:00 | 494.90 | 494.60 | 494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 494.90 | 494.66 | 494.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 497.30 | 494.66 | 494.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 495.70 | 494.87 | 494.32 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 493.50 | 494.24 | 494.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 490.40 | 493.48 | 493.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 490.25 | 489.15 | 490.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 490.25 | 489.15 | 490.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 490.25 | 489.15 | 490.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 490.25 | 489.15 | 490.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 491.40 | 489.60 | 490.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 492.00 | 489.60 | 490.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 495.90 | 490.86 | 491.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 496.75 | 490.86 | 491.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 491.90 | 491.07 | 491.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 491.20 | 491.07 | 491.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:15:00 | 491.30 | 491.25 | 491.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 492.00 | 491.40 | 491.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 492.00 | 491.40 | 491.39 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 491.00 | 491.32 | 491.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 15:15:00 | 490.75 | 491.21 | 491.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 493.90 | 487.08 | 488.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 493.90 | 487.08 | 488.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 493.90 | 487.08 | 488.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 493.90 | 487.08 | 488.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 489.95 | 487.66 | 488.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 487.25 | 487.35 | 488.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 486.45 | 485.19 | 485.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 13:15:00 | 462.89 | 467.56 | 472.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 13:15:00 | 462.13 | 467.56 | 472.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 466.65 | 464.04 | 469.62 | SL hit (close>ema200) qty=0.50 sl=464.04 alert=retest2 |

### Cycle 179 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 318.20 | 308.01 | 307.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 336.90 | 317.77 | 312.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 325.20 | 327.54 | 321.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 325.20 | 327.54 | 321.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 322.50 | 326.53 | 321.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 323.45 | 326.53 | 321.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 319.40 | 325.11 | 321.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 319.40 | 325.11 | 321.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 316.75 | 323.44 | 320.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:00:00 | 316.75 | 323.44 | 320.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 307.70 | 317.95 | 318.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 303.65 | 308.47 | 312.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 314.20 | 308.00 | 311.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 314.20 | 308.00 | 311.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 314.20 | 308.00 | 311.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 313.70 | 308.00 | 311.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 312.55 | 308.91 | 311.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 314.30 | 308.91 | 311.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 312.75 | 310.01 | 311.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:45:00 | 313.45 | 310.01 | 311.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 312.05 | 310.42 | 311.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:30:00 | 313.95 | 310.42 | 311.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 305.00 | 293.68 | 297.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 305.00 | 293.68 | 297.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 303.35 | 295.61 | 297.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 299.60 | 295.61 | 297.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:15:00 | 284.62 | 290.17 | 293.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 275.20 | 274.88 | 280.85 | SL hit (close>ema200) qty=0.50 sl=274.88 alert=retest2 |

### Cycle 181 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 247.20 | 244.52 | 244.35 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 240.95 | 243.94 | 244.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 240.60 | 242.41 | 243.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 242.00 | 238.70 | 240.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 15:15:00 | 242.00 | 238.70 | 240.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 242.00 | 238.70 | 240.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 263.60 | 238.70 | 240.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 257.85 | 242.53 | 241.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 268.15 | 252.08 | 246.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 293.65 | 296.69 | 280.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:30:00 | 292.20 | 296.69 | 280.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 286.80 | 290.80 | 285.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 288.55 | 290.80 | 285.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 283.60 | 288.48 | 285.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:45:00 | 283.35 | 288.48 | 285.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 282.60 | 287.30 | 285.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 282.60 | 287.30 | 285.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 279.20 | 283.89 | 284.28 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 287.20 | 284.42 | 284.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 293.90 | 286.31 | 284.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 300.00 | 300.34 | 296.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 303.85 | 300.34 | 296.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:00:00 | 302.50 | 300.77 | 297.16 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 300.05 | 300.41 | 297.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 297.85 | 300.41 | 297.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 297.95 | 299.62 | 297.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 297.50 | 299.62 | 297.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 297.70 | 299.23 | 297.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 297.70 | 299.23 | 297.71 | SL hit (close<ema400) qty=1.00 sl=297.71 alert=retest1 |

### Cycle 186 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 293.75 | 296.89 | 296.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 293.40 | 296.19 | 296.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 286.05 | 285.19 | 288.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 286.05 | 285.19 | 288.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 286.75 | 285.09 | 287.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 287.90 | 285.09 | 287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 286.80 | 285.58 | 287.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 287.20 | 285.58 | 287.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 286.60 | 285.78 | 287.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 287.20 | 285.78 | 287.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 285.70 | 285.77 | 286.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 286.60 | 285.77 | 286.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 305.30 | 289.69 | 288.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 340.90 | 310.01 | 300.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 324.35 | 326.77 | 314.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:30:00 | 324.25 | 326.77 | 314.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 313.55 | 324.67 | 320.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 313.55 | 324.67 | 320.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 313.00 | 322.33 | 320.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:45:00 | 307.20 | 322.33 | 320.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 315.65 | 318.54 | 318.61 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 319.45 | 318.72 | 318.69 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 311.40 | 317.26 | 318.03 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 321.90 | 318.70 | 318.34 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 313.80 | 318.03 | 318.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 288.65 | 311.05 | 314.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 271.50 | 269.70 | 276.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 271.50 | 269.70 | 276.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 245.40 | 245.41 | 249.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 244.00 | 245.41 | 249.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 242.05 | 244.77 | 248.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 15:15:00 | 231.80 | 235.50 | 240.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 235.90 | 235.58 | 239.79 | SL hit (close>ema200) qty=0.50 sl=235.58 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 241.00 | 237.88 | 237.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 243.00 | 238.90 | 238.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 234.60 | 238.04 | 237.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 234.60 | 238.04 | 237.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 234.60 | 238.04 | 237.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 234.70 | 238.04 | 237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 232.40 | 236.91 | 237.27 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 238.23 | 236.06 | 235.89 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 229.68 | 235.10 | 235.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 225.80 | 233.24 | 234.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 231.50 | 230.53 | 232.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 240.45 | 230.53 | 232.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 247.10 | 233.84 | 234.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:00:00 | 247.10 | 233.84 | 234.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 247.22 | 236.52 | 235.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 256.00 | 246.06 | 241.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 250.64 | 251.28 | 246.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:30:00 | 248.16 | 251.28 | 246.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 246.82 | 251.05 | 248.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 246.82 | 251.05 | 248.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 247.90 | 250.42 | 248.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 245.49 | 250.42 | 248.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 248.33 | 250.14 | 248.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 248.33 | 250.14 | 248.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 248.13 | 249.74 | 248.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 247.71 | 249.74 | 248.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 248.48 | 249.26 | 248.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 261.83 | 249.26 | 248.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 14:15:00 | 288.01 | 270.89 | 261.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 273.37 | 278.89 | 279.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 270.50 | 274.90 | 276.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 273.00 | 270.59 | 273.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 273.00 | 270.59 | 273.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 273.00 | 270.59 | 273.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 273.00 | 270.59 | 273.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 269.98 | 270.47 | 273.49 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 298.15 | 278.63 | 276.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 311.35 | 297.45 | 288.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 302.30 | 304.10 | 296.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 308.26 | 304.10 | 296.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 304.20 | 311.28 | 305.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 304.20 | 311.28 | 305.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 305.11 | 310.05 | 305.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 303.95 | 310.05 | 305.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 303.73 | 308.78 | 305.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 302.50 | 308.78 | 305.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 303.21 | 306.35 | 305.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 303.21 | 306.35 | 305.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 304.30 | 305.94 | 305.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 301.42 | 305.94 | 305.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 306.66 | 306.08 | 305.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 303.33 | 306.08 | 305.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 305.08 | 305.88 | 305.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 305.08 | 305.88 | 305.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 302.10 | 305.13 | 304.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 302.10 | 305.13 | 304.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 302.47 | 304.59 | 304.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 09:15:00 | 299.34 | 302.79 | 303.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 303.19 | 299.03 | 300.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 303.19 | 299.03 | 300.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 303.19 | 299.03 | 300.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 303.19 | 299.03 | 300.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 315.34 | 302.29 | 302.09 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 298.05 | 306.93 | 307.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 296.60 | 304.87 | 306.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 305.50 | 303.74 | 305.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 305.50 | 303.74 | 305.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 305.50 | 303.74 | 305.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 305.50 | 303.74 | 305.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 306.00 | 304.19 | 305.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 294.60 | 304.19 | 305.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 279.87 | 291.89 | 294.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 282.00 | 281.15 | 286.39 | SL hit (close>ema200) qty=0.50 sl=281.15 alert=retest2 |

### Cycle 203 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 288.55 | 287.22 | 287.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 293.00 | 290.29 | 289.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 280.15 | 288.85 | 288.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 280.15 | 288.85 | 288.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 280.15 | 288.85 | 288.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 280.15 | 288.85 | 288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 278.35 | 286.75 | 287.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 277.50 | 283.84 | 286.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 271.10 | 267.87 | 273.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 271.10 | 267.87 | 273.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 275.20 | 269.34 | 273.69 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 284.75 | 277.51 | 276.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 288.15 | 281.58 | 278.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 282.85 | 285.40 | 282.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 282.85 | 285.40 | 282.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 282.85 | 285.40 | 282.40 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 273.00 | 280.09 | 280.81 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 284.25 | 281.35 | 281.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 14:15:00 | 285.30 | 282.51 | 281.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 276.50 | 282.02 | 281.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 276.50 | 282.02 | 281.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 276.50 | 282.02 | 281.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:15:00 | 274.25 | 282.02 | 281.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 275.40 | 280.69 | 281.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 271.30 | 277.85 | 279.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 274.75 | 274.22 | 277.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 274.75 | 274.22 | 277.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 274.75 | 274.22 | 277.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 274.25 | 274.22 | 277.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 287.55 | 279.65 | 278.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 287.55 | 279.65 | 278.65 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 273.05 | 279.38 | 279.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 271.50 | 274.97 | 277.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 275.60 | 264.34 | 268.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 275.60 | 264.34 | 268.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 275.60 | 264.34 | 268.80 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 273.80 | 271.17 | 270.99 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 266.45 | 270.81 | 270.90 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 272.10 | 270.85 | 270.73 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 268.50 | 270.38 | 270.52 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 272.45 | 270.69 | 270.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 274.25 | 272.24 | 271.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 272.15 | 272.63 | 271.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 272.15 | 272.63 | 271.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 272.15 | 272.63 | 271.77 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 268.70 | 270.95 | 271.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 13:15:00 | 268.20 | 270.40 | 270.87 | Break + close below crossover candle low |

### Cycle 217 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 280.00 | 271.73 | 271.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 290.50 | 283.57 | 280.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 319.30 | 320.00 | 313.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:45:00 | 318.15 | 320.00 | 313.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 305.45 | 324.67 | 322.40 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 305.05 | 320.75 | 320.82 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 322.05 | 317.99 | 317.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 15:15:00 | 324.10 | 320.50 | 319.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 338.75 | 341.19 | 334.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 338.75 | 341.19 | 334.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 337.95 | 339.37 | 335.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 332.30 | 339.37 | 335.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 329.00 | 337.30 | 335.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 329.00 | 337.30 | 335.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 328.30 | 335.50 | 334.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 327.90 | 335.50 | 334.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 326.80 | 333.76 | 333.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 324.30 | 329.93 | 331.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 329.90 | 327.14 | 329.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 11:15:00 | 329.90 | 327.14 | 329.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 329.90 | 327.14 | 329.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 329.90 | 327.14 | 329.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 329.80 | 327.67 | 329.69 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 332.35 | 330.76 | 330.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 332.70 | 331.15 | 330.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 331.25 | 331.66 | 331.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 331.25 | 331.66 | 331.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 331.25 | 331.66 | 331.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 329.75 | 331.66 | 331.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 329.00 | 331.13 | 330.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 329.30 | 331.13 | 330.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 323.40 | 329.58 | 330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 318.45 | 325.15 | 327.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 321.90 | 317.57 | 321.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 321.90 | 317.57 | 321.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 321.90 | 317.57 | 321.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 321.90 | 317.57 | 321.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 320.00 | 318.05 | 321.56 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 327.65 | 324.07 | 323.66 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 256.50 | 2024-04-15 09:15:00 | 282.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-13 10:15:00 | 279.45 | 2024-05-13 14:15:00 | 302.00 | STOP_HIT | 1.00 | -8.07% |
| BUY | retest2 | 2024-05-16 11:45:00 | 314.98 | 2024-05-21 13:15:00 | 309.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-05-17 11:00:00 | 313.00 | 2024-05-21 13:15:00 | 309.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-05-17 11:45:00 | 313.25 | 2024-05-21 13:15:00 | 309.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-05-17 12:15:00 | 314.98 | 2024-05-21 13:15:00 | 309.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-05-29 13:15:00 | 325.40 | 2024-06-03 09:15:00 | 357.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-24 09:15:00 | 376.98 | 2024-06-27 14:15:00 | 358.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 09:15:00 | 376.98 | 2024-06-28 15:15:00 | 364.05 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest2 | 2024-07-04 09:15:00 | 400.03 | 2024-07-08 09:15:00 | 390.03 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-07-04 13:45:00 | 400.53 | 2024-07-08 09:15:00 | 390.03 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-07-04 14:30:00 | 400.00 | 2024-07-08 09:15:00 | 390.03 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest1 | 2024-07-11 15:00:00 | 371.80 | 2024-07-12 12:15:00 | 385.50 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest1 | 2024-07-12 09:30:00 | 371.50 | 2024-07-12 12:15:00 | 385.50 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2024-07-22 12:00:00 | 362.00 | 2024-07-24 09:15:00 | 375.50 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2024-07-23 09:15:00 | 357.23 | 2024-07-24 09:15:00 | 375.50 | STOP_HIT | 1.00 | -5.11% |
| SELL | retest2 | 2024-07-23 15:00:00 | 360.20 | 2024-07-24 09:15:00 | 375.50 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-08-06 13:30:00 | 352.50 | 2024-08-12 09:15:00 | 334.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 11:00:00 | 355.00 | 2024-08-12 09:15:00 | 337.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 12:15:00 | 354.50 | 2024-08-12 09:15:00 | 336.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 13:30:00 | 355.18 | 2024-08-12 09:15:00 | 337.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 13:30:00 | 352.50 | 2024-08-12 13:15:00 | 344.98 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2024-08-07 11:00:00 | 355.00 | 2024-08-12 13:15:00 | 344.98 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2024-08-07 12:15:00 | 354.50 | 2024-08-12 13:15:00 | 344.98 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2024-08-07 13:30:00 | 355.18 | 2024-08-12 13:15:00 | 344.98 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2024-08-09 10:30:00 | 344.50 | 2024-08-16 09:15:00 | 345.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-08-09 11:30:00 | 346.50 | 2024-08-16 09:15:00 | 345.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2024-08-09 12:00:00 | 345.65 | 2024-08-16 12:15:00 | 346.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-08-09 13:15:00 | 345.00 | 2024-08-16 12:15:00 | 346.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-08-13 14:45:00 | 339.03 | 2024-08-16 12:15:00 | 346.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-08-14 14:45:00 | 337.50 | 2024-08-16 12:15:00 | 346.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-08-29 11:15:00 | 355.55 | 2024-08-30 14:15:00 | 372.60 | STOP_HIT | 1.00 | -4.80% |
| SELL | retest2 | 2024-08-29 13:15:00 | 355.03 | 2024-08-30 14:15:00 | 372.60 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2024-09-10 09:15:00 | 348.75 | 2024-09-12 10:15:00 | 331.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-10 09:15:00 | 348.75 | 2024-09-13 09:15:00 | 336.50 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2024-11-04 12:45:00 | 457.40 | 2024-11-07 12:15:00 | 503.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 13:30:00 | 494.50 | 2024-12-03 14:15:00 | 489.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-11-29 14:00:00 | 497.00 | 2024-12-03 14:15:00 | 489.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-02 10:45:00 | 494.50 | 2024-12-03 14:15:00 | 489.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-05 15:15:00 | 515.00 | 2024-12-12 09:15:00 | 566.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-06 11:00:00 | 515.25 | 2024-12-12 09:15:00 | 566.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-10 13:45:00 | 519.25 | 2024-12-16 10:15:00 | 571.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-08 14:00:00 | 647.50 | 2025-01-09 09:15:00 | 588.10 | STOP_HIT | 1.00 | -9.17% |
| SELL | retest2 | 2025-01-24 13:45:00 | 473.33 | 2025-01-28 09:15:00 | 449.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:45:00 | 473.33 | 2025-01-28 12:15:00 | 449.50 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2025-01-31 14:00:00 | 435.00 | 2025-02-03 09:15:00 | 413.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 09:15:00 | 436.03 | 2025-02-03 09:15:00 | 414.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 09:45:00 | 436.00 | 2025-02-03 09:15:00 | 414.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 10:15:00 | 433.80 | 2025-02-03 09:15:00 | 412.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 14:00:00 | 435.00 | 2025-02-04 09:15:00 | 391.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 09:15:00 | 436.03 | 2025-02-04 09:15:00 | 392.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 09:45:00 | 436.00 | 2025-02-04 09:15:00 | 392.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 10:15:00 | 433.80 | 2025-02-04 09:15:00 | 390.42 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-10 14:30:00 | 433.98 | 2025-02-12 09:15:00 | 408.23 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest2 | 2025-02-27 09:15:00 | 433.00 | 2025-02-27 12:15:00 | 410.50 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2025-02-27 09:45:00 | 432.65 | 2025-02-27 12:15:00 | 410.50 | STOP_HIT | 1.00 | -5.12% |
| BUY | retest2 | 2025-02-27 10:45:00 | 428.45 | 2025-02-27 12:15:00 | 410.50 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-03-04 11:30:00 | 379.45 | 2025-03-05 09:15:00 | 409.10 | STOP_HIT | 1.00 | -7.81% |
| SELL | retest2 | 2025-03-04 13:30:00 | 378.50 | 2025-03-05 09:15:00 | 409.10 | STOP_HIT | 1.00 | -8.08% |
| SELL | retest2 | 2025-03-12 11:15:00 | 381.05 | 2025-03-17 13:15:00 | 388.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-03-13 09:15:00 | 371.10 | 2025-03-17 13:15:00 | 388.80 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-05-02 11:30:00 | 496.70 | 2025-05-05 14:15:00 | 508.80 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-05-02 14:15:00 | 495.15 | 2025-05-05 14:15:00 | 508.80 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-05-28 09:30:00 | 525.75 | 2025-05-28 14:15:00 | 511.35 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-05-28 10:15:00 | 522.90 | 2025-05-28 14:15:00 | 511.35 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-02 10:30:00 | 535.00 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-03 09:15:00 | 536.60 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-06-04 09:15:00 | 536.10 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-04 11:30:00 | 539.10 | 2025-06-06 09:15:00 | 524.05 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-06-10 12:15:00 | 517.95 | 2025-06-13 09:15:00 | 492.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 09:45:00 | 518.20 | 2025-06-13 09:15:00 | 492.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 12:15:00 | 517.95 | 2025-06-16 14:15:00 | 494.95 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2025-06-11 09:45:00 | 518.20 | 2025-06-16 14:15:00 | 494.95 | STOP_HIT | 0.50 | 4.49% |
| BUY | retest2 | 2025-06-30 09:15:00 | 499.15 | 2025-06-30 15:15:00 | 488.80 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-07-04 11:15:00 | 469.05 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-07-07 09:30:00 | 468.30 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-07-07 10:15:00 | 469.20 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-07-07 10:45:00 | 469.10 | 2025-07-07 11:15:00 | 482.15 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-07-10 12:15:00 | 514.30 | 2025-07-11 09:15:00 | 501.05 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-07-15 15:00:00 | 513.00 | 2025-07-21 09:15:00 | 511.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-07-16 10:15:00 | 517.50 | 2025-07-21 09:15:00 | 511.55 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest1 | 2025-08-01 11:30:00 | 519.50 | 2025-08-04 12:15:00 | 545.48 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-08-01 11:30:00 | 519.50 | 2025-08-05 09:15:00 | 571.45 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-04 12:45:00 | 532.25 | 2025-08-06 09:15:00 | 508.00 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-08-08 10:45:00 | 499.75 | 2025-08-11 10:15:00 | 529.70 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2025-08-18 12:00:00 | 505.10 | 2025-08-25 11:15:00 | 510.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-29 12:15:00 | 494.85 | 2025-09-01 10:15:00 | 501.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-01 10:00:00 | 494.70 | 2025-09-01 10:15:00 | 501.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-01 10:30:00 | 495.40 | 2025-09-01 11:15:00 | 504.15 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-08 09:15:00 | 499.20 | 2025-09-10 09:15:00 | 505.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-19 09:15:00 | 526.50 | 2025-09-24 11:15:00 | 523.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-10 11:15:00 | 491.20 | 2025-10-10 13:15:00 | 492.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-10-10 13:15:00 | 491.30 | 2025-10-10 13:15:00 | 492.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-10-14 09:45:00 | 487.25 | 2025-10-28 13:15:00 | 462.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 486.45 | 2025-10-28 13:15:00 | 462.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-14 09:45:00 | 487.25 | 2025-10-29 09:15:00 | 466.65 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-10-16 09:30:00 | 486.45 | 2025-10-29 09:15:00 | 466.65 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-11-25 14:15:00 | 299.60 | 2025-11-27 11:15:00 | 284.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 14:15:00 | 299.60 | 2025-12-01 09:15:00 | 275.20 | STOP_HIT | 0.50 | 8.14% |
| BUY | retest1 | 2025-12-24 09:15:00 | 303.85 | 2025-12-24 14:15:00 | 297.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-12-24 10:00:00 | 302.50 | 2025-12-24 14:15:00 | 297.70 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-01-22 10:15:00 | 244.00 | 2026-01-23 15:15:00 | 231.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 244.00 | 2026-01-27 09:15:00 | 235.90 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2026-01-22 10:45:00 | 242.05 | 2026-01-27 09:15:00 | 229.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 242.05 | 2026-01-27 09:15:00 | 235.90 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest2 | 2026-02-09 09:15:00 | 261.83 | 2026-02-09 14:15:00 | 288.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 294.60 | 2026-03-09 09:15:00 | 279.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 294.60 | 2026-03-10 09:15:00 | 282.00 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2026-03-24 10:15:00 | 274.25 | 2026-03-25 09:15:00 | 287.55 | STOP_HIT | 1.00 | -4.85% |
