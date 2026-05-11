# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 125.09
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 149 |
| ALERT2 | 146 |
| ALERT2_SKIP | 88 |
| ALERT3 | 320 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 108 |
| PARTIAL | 14 |
| TARGET_HIT | 8 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 77
- **Target hits / Stop hits / Partials:** 8 / 102 / 14
- **Avg / median % per leg:** 0.31% / -1.00%
- **Sum % (uncompounded):** 38.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 14 | 27.5% | 8 | 43 | 0 | 0.41% | 20.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 51 | 14 | 27.5% | 8 | 43 | 0 | 0.41% | 20.8% |
| SELL (all) | 73 | 33 | 45.2% | 0 | 59 | 14 | 0.24% | 17.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.87% | -7.7% |
| SELL @ 3rd Alert (retest2) | 71 | 33 | 46.5% | 0 | 57 | 14 | 0.35% | 25.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.87% | -7.7% |
| retest2 (combined) | 122 | 47 | 38.5% | 8 | 100 | 14 | 0.38% | 45.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 10:15:00 | 55.75 | 55.46 | 55.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 09:15:00 | 56.85 | 55.84 | 55.64 | Break + close above crossover candle high |

### Cycle 2 — SELL (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 13:15:00 | 53.45 | 55.61 | 55.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 52.45 | 53.32 | 54.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 53.35 | 53.14 | 53.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 10:15:00 | 53.35 | 53.07 | 53.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 53.35 | 53.07 | 53.51 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 53.95 | 52.39 | 52.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 15:15:00 | 54.30 | 52.77 | 52.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 13:15:00 | 54.50 | 54.51 | 53.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 10:15:00 | 54.30 | 54.36 | 54.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 54.30 | 54.36 | 54.03 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 56.85 | 57.20 | 57.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 09:15:00 | 56.55 | 56.98 | 57.11 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 58.85 | 57.13 | 57.08 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 09:15:00 | 57.30 | 57.68 | 57.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 11:15:00 | 56.95 | 57.49 | 57.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 11:15:00 | 57.25 | 57.07 | 57.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 11:15:00 | 57.25 | 57.07 | 57.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 57.25 | 57.07 | 57.28 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 11:15:00 | 58.15 | 57.32 | 57.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 12:15:00 | 58.95 | 58.31 | 57.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 13:15:00 | 58.25 | 58.29 | 57.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 14:15:00 | 57.95 | 58.23 | 57.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 57.95 | 58.23 | 57.92 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 09:15:00 | 61.85 | 62.48 | 62.49 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 11:15:00 | 63.60 | 62.61 | 62.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 65.70 | 63.45 | 63.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 64.70 | 64.87 | 64.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 64.15 | 64.73 | 64.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 64.15 | 64.73 | 64.15 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 62.85 | 63.84 | 63.96 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 15:15:00 | 64.50 | 63.62 | 63.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 65.50 | 64.00 | 63.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 63.40 | 64.43 | 64.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 63.40 | 64.43 | 64.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 63.40 | 64.43 | 64.06 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 63.00 | 63.74 | 63.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 12:15:00 | 62.85 | 63.18 | 63.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 61.60 | 61.52 | 62.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 61.60 | 61.52 | 62.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 61.60 | 61.52 | 62.00 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 12:15:00 | 63.10 | 62.02 | 61.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 15:15:00 | 63.25 | 62.58 | 62.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 12:15:00 | 63.15 | 63.16 | 62.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 15:15:00 | 63.05 | 63.13 | 62.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 63.05 | 63.13 | 62.92 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 11:15:00 | 63.30 | 63.70 | 63.74 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 64.55 | 63.82 | 63.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 11:15:00 | 64.95 | 64.05 | 63.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 66.45 | 67.03 | 66.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 66.45 | 67.03 | 66.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 66.45 | 67.03 | 66.10 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 62.90 | 65.48 | 65.70 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 65.95 | 64.93 | 64.82 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 64.85 | 65.37 | 65.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 10:15:00 | 64.20 | 64.75 | 65.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 13:15:00 | 63.15 | 63.03 | 63.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 63.20 | 63.05 | 63.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 63.20 | 63.05 | 63.38 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 65.55 | 63.24 | 63.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 66.20 | 63.83 | 63.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 12:15:00 | 68.85 | 69.17 | 67.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 68.25 | 68.78 | 68.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 68.25 | 68.78 | 68.03 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 12:15:00 | 72.00 | 72.60 | 72.65 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 11:15:00 | 73.55 | 72.77 | 72.70 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 69.85 | 72.33 | 72.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 68.70 | 71.24 | 72.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 13:15:00 | 67.95 | 67.64 | 69.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 69.70 | 68.34 | 69.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 69.70 | 68.34 | 69.13 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 71.60 | 69.80 | 69.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 13:15:00 | 72.15 | 70.27 | 69.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 71.10 | 71.42 | 70.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 70.90 | 71.30 | 70.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 70.90 | 71.30 | 70.83 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 12:15:00 | 71.70 | 72.14 | 72.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 13:15:00 | 71.65 | 72.04 | 72.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 72.30 | 71.86 | 71.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 72.30 | 71.86 | 71.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 72.30 | 71.86 | 71.98 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 12:15:00 | 72.25 | 72.08 | 72.07 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 13:15:00 | 71.60 | 71.98 | 72.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 09:15:00 | 71.40 | 71.73 | 71.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 09:15:00 | 71.75 | 71.35 | 71.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 71.75 | 71.35 | 71.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 71.75 | 71.35 | 71.54 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 71.85 | 71.57 | 71.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 09:15:00 | 73.00 | 71.99 | 71.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 11:15:00 | 71.95 | 72.08 | 71.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 12:15:00 | 71.95 | 72.05 | 71.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 71.95 | 72.05 | 71.85 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 71.25 | 71.77 | 71.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 70.55 | 71.52 | 71.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 10:15:00 | 70.95 | 70.89 | 71.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 69.95 | 70.49 | 70.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 69.95 | 70.49 | 70.86 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 13:15:00 | 67.60 | 67.50 | 67.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 15:15:00 | 68.00 | 67.65 | 67.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 67.80 | 67.82 | 67.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 14:15:00 | 67.80 | 67.82 | 67.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 67.80 | 67.82 | 67.70 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 67.80 | 68.09 | 68.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 66.90 | 67.85 | 67.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 10:15:00 | 67.45 | 67.20 | 67.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 66.90 | 67.14 | 67.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 66.90 | 67.14 | 67.44 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 62.05 | 61.08 | 61.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 09:15:00 | 63.90 | 61.99 | 61.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 14:15:00 | 63.30 | 63.30 | 62.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 12:15:00 | 63.10 | 63.41 | 62.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 63.10 | 63.41 | 62.87 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 12:15:00 | 63.35 | 63.61 | 63.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 10:15:00 | 63.15 | 63.41 | 63.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 63.00 | 62.88 | 63.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 63.00 | 62.88 | 63.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 63.00 | 62.88 | 63.06 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 63.80 | 63.09 | 63.07 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 09:15:00 | 62.95 | 63.12 | 63.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-15 14:15:00 | 62.30 | 62.85 | 62.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 62.80 | 62.77 | 62.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 62.80 | 62.77 | 62.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 62.80 | 62.77 | 62.93 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 11:15:00 | 63.30 | 62.80 | 62.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 10:15:00 | 64.60 | 63.19 | 62.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 14:15:00 | 64.00 | 64.31 | 63.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 14:15:00 | 64.00 | 64.31 | 63.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 64.00 | 64.31 | 63.91 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 10:15:00 | 64.70 | 65.30 | 65.33 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 10:15:00 | 65.95 | 65.32 | 65.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 11:15:00 | 67.15 | 65.68 | 65.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 15:15:00 | 66.00 | 66.20 | 65.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 67.80 | 66.52 | 66.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 67.80 | 66.52 | 66.00 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 13:15:00 | 70.75 | 71.57 | 71.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 69.80 | 70.59 | 70.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 69.25 | 68.05 | 68.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 69.25 | 68.05 | 68.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 69.25 | 68.05 | 68.80 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 69.55 | 69.11 | 69.06 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 14:15:00 | 68.95 | 69.10 | 69.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 15:15:00 | 68.55 | 68.99 | 69.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-28 11:15:00 | 68.70 | 68.62 | 68.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 11:15:00 | 68.70 | 68.62 | 68.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 68.70 | 68.62 | 68.85 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 11:15:00 | 69.35 | 68.59 | 68.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 13:15:00 | 71.25 | 69.27 | 68.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 13:15:00 | 80.45 | 80.52 | 77.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 14:15:00 | 78.35 | 79.69 | 78.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 78.35 | 79.69 | 78.92 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 77.55 | 78.66 | 78.66 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 10:15:00 | 79.10 | 78.53 | 78.46 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 78.00 | 78.42 | 78.42 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 15:15:00 | 78.70 | 78.40 | 78.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 79.60 | 78.64 | 78.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 78.70 | 78.71 | 78.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 14:15:00 | 78.75 | 78.71 | 78.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 78.75 | 78.71 | 78.60 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 15:15:00 | 78.20 | 78.54 | 78.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 10:15:00 | 77.40 | 78.24 | 78.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 78.50 | 77.92 | 78.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 78.50 | 77.92 | 78.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 78.50 | 77.92 | 78.13 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 10:15:00 | 79.80 | 77.11 | 76.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 11:15:00 | 81.35 | 77.96 | 77.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 10:15:00 | 80.15 | 80.22 | 78.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 11:15:00 | 79.35 | 80.05 | 79.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 79.35 | 80.05 | 79.01 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 77.65 | 78.54 | 78.57 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 80.00 | 78.70 | 78.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 81.55 | 79.49 | 78.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 09:15:00 | 80.25 | 80.56 | 79.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 11:15:00 | 80.35 | 80.48 | 80.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 80.35 | 80.48 | 80.00 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 12:15:00 | 79.30 | 79.81 | 79.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 78.60 | 79.51 | 79.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 80.00 | 79.49 | 79.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 80.00 | 79.49 | 79.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 80.00 | 79.49 | 79.67 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 80.20 | 79.86 | 79.83 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 10:15:00 | 79.00 | 79.66 | 79.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 11:15:00 | 78.30 | 79.39 | 79.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 10:15:00 | 78.20 | 76.83 | 77.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 10:15:00 | 78.20 | 76.83 | 77.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 78.20 | 76.83 | 77.60 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 14:15:00 | 78.45 | 78.07 | 78.03 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 09:15:00 | 74.40 | 77.33 | 77.70 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 11:15:00 | 78.25 | 76.48 | 76.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 12:15:00 | 83.00 | 77.78 | 76.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 80.10 | 80.15 | 78.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 80.10 | 80.15 | 78.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 80.10 | 80.15 | 78.53 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 76.90 | 79.16 | 79.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 76.55 | 78.21 | 78.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 77.55 | 77.49 | 78.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 78.00 | 77.59 | 78.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 78.00 | 77.59 | 78.06 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 79.80 | 78.21 | 78.11 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 09:15:00 | 77.80 | 78.69 | 78.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 10:15:00 | 77.40 | 77.83 | 78.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 79.15 | 77.68 | 77.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 79.15 | 77.68 | 77.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 79.15 | 77.68 | 77.90 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 11:15:00 | 80.25 | 78.34 | 78.17 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 76.70 | 77.87 | 78.02 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 80.20 | 78.05 | 77.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 14:15:00 | 80.90 | 79.40 | 78.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 79.55 | 79.64 | 78.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 11:15:00 | 78.95 | 79.45 | 78.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 78.95 | 79.45 | 78.94 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 10:15:00 | 78.05 | 78.81 | 78.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 11:15:00 | 77.70 | 78.59 | 78.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 76.25 | 76.19 | 76.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 77.25 | 76.37 | 76.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 77.25 | 76.37 | 76.72 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 77.05 | 76.90 | 76.88 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 76.70 | 76.86 | 76.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 10:15:00 | 76.40 | 76.77 | 76.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 10:15:00 | 76.05 | 75.95 | 76.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 10:15:00 | 76.05 | 75.95 | 76.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 76.05 | 75.95 | 76.32 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 68.20 | 67.94 | 67.91 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 09:15:00 | 67.45 | 67.87 | 67.89 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 12:15:00 | 68.05 | 67.70 | 67.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 09:15:00 | 68.45 | 67.98 | 67.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 10:15:00 | 67.80 | 67.95 | 67.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 10:15:00 | 67.80 | 67.95 | 67.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 67.80 | 67.95 | 67.84 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 12:15:00 | 67.30 | 67.78 | 67.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 66.35 | 67.41 | 67.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 68.45 | 67.47 | 67.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 68.45 | 67.47 | 67.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 68.45 | 67.47 | 67.59 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 68.65 | 67.71 | 67.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 68.95 | 67.96 | 67.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 73.20 | 73.22 | 72.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 73.20 | 73.22 | 72.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 73.20 | 73.22 | 72.27 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 12:15:00 | 72.20 | 73.03 | 73.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 71.95 | 72.81 | 72.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 11:15:00 | 72.70 | 72.45 | 72.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 11:15:00 | 72.70 | 72.45 | 72.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 72.70 | 72.45 | 72.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 72.75 | 72.56 | 72.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 72.05 | 72.46 | 72.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:15:00 | 71.75 | 72.28 | 72.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 13:15:00 | 71.70 | 72.19 | 72.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 68.16 | 71.39 | 71.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 68.11 | 71.39 | 71.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 15:15:00 | 70.00 | 69.82 | 70.36 | SL hit (close>ema200) qty=0.50 sl=69.82 alert=retest2 |

### Cycle 71 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 69.55 | 69.43 | 69.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 70.15 | 69.61 | 69.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 10:15:00 | 71.50 | 71.52 | 70.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:45:00 | 71.40 | 71.52 | 70.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 71.15 | 71.48 | 71.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:45:00 | 71.10 | 71.48 | 71.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 71.20 | 71.43 | 71.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 71.60 | 71.43 | 71.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 71.00 | 71.34 | 71.08 | SL hit (close<static) qty=1.00 sl=71.05 alert=retest2 |

### Cycle 72 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 70.35 | 71.03 | 71.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 69.75 | 70.41 | 70.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 15:15:00 | 68.40 | 68.23 | 68.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 09:15:00 | 67.45 | 68.23 | 68.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 66.70 | 67.06 | 67.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 66.60 | 67.06 | 67.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 13:15:00 | 66.50 | 65.67 | 65.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 66.50 | 65.67 | 65.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 15:15:00 | 66.65 | 66.00 | 65.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 12:15:00 | 69.85 | 70.21 | 69.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 13:00:00 | 69.85 | 70.21 | 69.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 69.20 | 69.96 | 69.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:30:00 | 69.25 | 69.96 | 69.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 69.40 | 69.85 | 69.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 70.45 | 69.85 | 69.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:45:00 | 69.70 | 69.95 | 69.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:45:00 | 69.70 | 69.66 | 69.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 13:30:00 | 70.70 | 69.76 | 69.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 70.10 | 69.96 | 69.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 69.55 | 69.96 | 69.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 69.75 | 69.93 | 69.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 69.75 | 69.93 | 69.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 69.65 | 69.87 | 69.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 69.65 | 69.87 | 69.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 69.90 | 69.88 | 69.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 69.75 | 69.88 | 69.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 69.50 | 69.80 | 69.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 69.65 | 69.80 | 69.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 69.15 | 69.67 | 69.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 69.15 | 69.67 | 69.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 10:15:00 | 68.45 | 68.90 | 69.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 12:15:00 | 68.95 | 68.85 | 69.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 12:15:00 | 68.95 | 68.85 | 69.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 68.95 | 68.85 | 69.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:45:00 | 68.95 | 68.85 | 69.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 68.55 | 68.79 | 68.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:15:00 | 68.20 | 68.70 | 68.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 13:30:00 | 68.25 | 68.56 | 68.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 13:15:00 | 71.10 | 69.15 | 68.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 71.10 | 69.15 | 68.93 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 67.35 | 69.95 | 70.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 65.35 | 69.03 | 69.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 66.85 | 66.77 | 67.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:30:00 | 66.70 | 66.77 | 67.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 67.65 | 67.03 | 67.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 67.65 | 67.03 | 67.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 67.35 | 67.09 | 67.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 68.45 | 67.09 | 67.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 68.80 | 67.43 | 67.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 69.15 | 67.43 | 67.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 69.15 | 67.78 | 67.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 69.45 | 67.78 | 67.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 67.55 | 67.77 | 67.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:30:00 | 68.45 | 67.77 | 67.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 67.30 | 67.67 | 67.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 67.95 | 67.67 | 67.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 68.60 | 67.83 | 67.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 68.60 | 67.83 | 67.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 68.50 | 67.97 | 67.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 69.72 | 68.48 | 68.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 15:15:00 | 75.60 | 75.61 | 74.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:15:00 | 74.90 | 75.61 | 74.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 75.12 | 75.22 | 74.46 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 73.74 | 74.30 | 74.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 73.45 | 74.13 | 74.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 10:15:00 | 74.41 | 74.03 | 74.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 10:15:00 | 74.41 | 74.03 | 74.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 74.41 | 74.03 | 74.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 74.41 | 74.03 | 74.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 74.15 | 74.06 | 74.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 74.02 | 74.06 | 74.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 74.21 | 74.09 | 74.19 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 76.00 | 74.60 | 74.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 78.60 | 75.66 | 74.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 87.05 | 87.07 | 82.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 09:45:00 | 87.05 | 87.07 | 82.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 84.52 | 85.85 | 84.29 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 83.06 | 83.97 | 84.02 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 84.19 | 83.67 | 83.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 84.82 | 83.90 | 83.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 83.74 | 84.26 | 83.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 83.74 | 84.26 | 83.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 83.74 | 84.26 | 83.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 83.85 | 84.26 | 83.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 83.32 | 84.07 | 83.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:45:00 | 83.40 | 84.07 | 83.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 82.50 | 83.76 | 83.80 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 85.75 | 83.85 | 83.67 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 83.78 | 84.06 | 84.09 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 86.41 | 84.53 | 84.30 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 83.50 | 84.23 | 84.27 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 86.79 | 84.60 | 84.42 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 84.34 | 84.68 | 84.69 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 86.24 | 84.99 | 84.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 12:15:00 | 88.13 | 85.97 | 85.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 15:15:00 | 89.60 | 90.32 | 88.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 88.80 | 90.32 | 88.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 87.14 | 89.69 | 88.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 87.14 | 89.69 | 88.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 87.10 | 89.17 | 88.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 86.07 | 89.17 | 88.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 87.99 | 88.78 | 88.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:15:00 | 88.04 | 88.78 | 88.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 87.71 | 88.56 | 88.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:00:00 | 87.71 | 88.56 | 88.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 87.80 | 88.41 | 88.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:45:00 | 87.71 | 88.41 | 88.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 15:15:00 | 87.31 | 88.19 | 88.22 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 89.75 | 88.45 | 88.32 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 87.16 | 88.12 | 88.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 14:15:00 | 86.43 | 87.66 | 87.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 11:15:00 | 88.96 | 87.57 | 87.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 11:15:00 | 88.96 | 87.57 | 87.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 88.96 | 87.57 | 87.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:30:00 | 89.22 | 87.57 | 87.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 88.52 | 87.76 | 87.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 13:45:00 | 88.08 | 87.87 | 87.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 14:15:00 | 88.33 | 87.96 | 87.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 88.33 | 87.96 | 87.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 93.84 | 89.21 | 88.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 14:15:00 | 90.74 | 90.99 | 89.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 15:00:00 | 90.74 | 90.99 | 89.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 91.55 | 91.08 | 90.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 90.30 | 91.08 | 90.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 89.71 | 90.80 | 90.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:45:00 | 89.72 | 90.80 | 90.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 90.53 | 90.75 | 90.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:15:00 | 90.75 | 90.72 | 90.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 88.01 | 90.00 | 89.97 | SL hit (close<static) qty=1.00 sl=89.55 alert=retest2 |

### Cycle 94 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 87.43 | 89.49 | 89.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 87.01 | 88.42 | 89.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 88.45 | 87.86 | 88.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 88.45 | 87.86 | 88.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 89.80 | 88.25 | 88.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 89.80 | 88.25 | 88.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 88.53 | 88.30 | 88.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 88.27 | 88.30 | 88.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 88.29 | 88.34 | 88.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:30:00 | 88.21 | 88.14 | 88.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:15:00 | 83.86 | 87.08 | 87.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:15:00 | 83.88 | 87.08 | 87.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 83.80 | 86.35 | 87.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 11:15:00 | 85.21 | 85.06 | 86.17 | SL hit (close>ema200) qty=0.50 sl=85.06 alert=retest2 |

### Cycle 95 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 87.35 | 85.24 | 85.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 14:15:00 | 94.09 | 88.21 | 86.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 91.91 | 92.17 | 90.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 91.91 | 92.17 | 90.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 90.53 | 91.69 | 90.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 91.23 | 91.69 | 90.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 90.47 | 91.44 | 90.51 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 88.39 | 89.97 | 90.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 86.80 | 88.38 | 89.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 09:15:00 | 82.65 | 82.33 | 83.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 10:15:00 | 83.67 | 82.60 | 83.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 83.67 | 82.60 | 83.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 83.67 | 82.60 | 83.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 84.15 | 82.91 | 83.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 84.15 | 82.91 | 83.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 84.23 | 83.18 | 83.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:45:00 | 84.80 | 83.18 | 83.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 82.95 | 83.18 | 83.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 83.41 | 83.18 | 83.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 84.16 | 83.33 | 83.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:45:00 | 84.55 | 83.33 | 83.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 83.34 | 83.44 | 83.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 83.34 | 83.44 | 83.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 14:15:00 | 88.25 | 84.40 | 83.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 88.45 | 85.72 | 84.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 86.84 | 87.89 | 86.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 86.84 | 87.89 | 86.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 86.84 | 87.89 | 86.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 86.95 | 87.89 | 86.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 87.21 | 87.76 | 86.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 87.02 | 87.76 | 86.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 86.75 | 87.40 | 86.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 86.62 | 87.40 | 86.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 86.35 | 87.19 | 86.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 86.35 | 87.19 | 86.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 85.69 | 86.89 | 86.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 85.69 | 86.89 | 86.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 83.96 | 85.89 | 86.15 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 85.80 | 85.43 | 85.41 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 85.23 | 85.37 | 85.39 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 14:15:00 | 85.92 | 85.48 | 85.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 86.85 | 85.83 | 85.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 86.05 | 86.95 | 86.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 86.05 | 86.95 | 86.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 86.05 | 86.95 | 86.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 86.05 | 86.95 | 86.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 86.57 | 86.87 | 86.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 87.75 | 86.87 | 86.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 86.48 | 88.24 | 88.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 86.48 | 88.24 | 88.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 86.40 | 87.87 | 88.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 87.70 | 87.12 | 87.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 87.70 | 87.12 | 87.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 87.70 | 87.12 | 87.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 87.99 | 87.12 | 87.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 88.65 | 87.43 | 87.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 88.65 | 87.43 | 87.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 88.11 | 87.56 | 87.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 87.73 | 87.78 | 87.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:00:00 | 87.95 | 87.78 | 87.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 10:15:00 | 87.92 | 87.81 | 87.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 87.92 | 87.81 | 87.81 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 87.35 | 87.72 | 87.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 87.07 | 87.59 | 87.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 86.35 | 85.79 | 86.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 86.35 | 85.79 | 86.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 86.35 | 85.79 | 86.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 85.50 | 85.73 | 86.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:30:00 | 85.50 | 85.67 | 86.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 85.38 | 84.24 | 84.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 11:15:00 | 85.40 | 84.47 | 84.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 85.40 | 84.47 | 84.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 14:15:00 | 85.88 | 85.02 | 84.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 84.89 | 85.07 | 84.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 84.89 | 85.07 | 84.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 84.89 | 85.07 | 84.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 84.89 | 85.07 | 84.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 85.50 | 85.16 | 84.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:15:00 | 86.10 | 85.16 | 84.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 14:30:00 | 86.51 | 85.92 | 85.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:45:00 | 86.20 | 86.08 | 85.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 87.75 | 85.71 | 85.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 86.10 | 86.99 | 86.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 86.10 | 86.99 | 86.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 84.92 | 86.58 | 86.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 84.92 | 86.58 | 86.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-11 15:15:00 | 85.34 | 86.33 | 86.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 85.34 | 86.33 | 86.37 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 87.64 | 86.08 | 85.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 87.79 | 87.24 | 86.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 86.88 | 87.29 | 86.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 86.88 | 87.29 | 86.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 86.88 | 87.29 | 86.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 86.88 | 87.29 | 86.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 86.90 | 87.21 | 86.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 86.41 | 87.21 | 86.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 86.19 | 87.01 | 86.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 86.32 | 87.01 | 86.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 85.75 | 86.76 | 86.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 84.57 | 86.04 | 86.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 83.90 | 83.22 | 83.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 83.90 | 83.22 | 83.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 83.90 | 83.22 | 83.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 84.00 | 83.22 | 83.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 84.14 | 83.40 | 83.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 84.14 | 83.40 | 83.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 84.44 | 83.61 | 83.99 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 15:15:00 | 84.80 | 84.30 | 84.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 85.64 | 84.70 | 84.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 84.00 | 84.76 | 84.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 84.00 | 84.76 | 84.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 84.00 | 84.76 | 84.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 84.00 | 84.76 | 84.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 84.69 | 84.75 | 84.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 84.64 | 84.75 | 84.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 84.02 | 84.60 | 84.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 84.05 | 84.60 | 84.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 83.95 | 84.47 | 84.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 83.75 | 84.47 | 84.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 84.61 | 84.51 | 84.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:15:00 | 84.60 | 84.51 | 84.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 84.60 | 84.53 | 84.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 84.50 | 84.53 | 84.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 84.06 | 84.43 | 84.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 14:15:00 | 83.70 | 84.13 | 84.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 84.52 | 84.12 | 84.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 84.52 | 84.12 | 84.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 84.52 | 84.12 | 84.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 84.52 | 84.12 | 84.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 84.49 | 84.20 | 84.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:45:00 | 84.25 | 84.15 | 84.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:30:00 | 84.11 | 83.63 | 83.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 86.98 | 84.30 | 83.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 86.98 | 84.30 | 83.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 88.06 | 85.05 | 84.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 87.34 | 87.55 | 86.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 14:00:00 | 87.34 | 87.55 | 86.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 85.80 | 87.20 | 86.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 86.00 | 87.20 | 86.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 86.60 | 87.08 | 86.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 83.96 | 87.08 | 86.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 86.12 | 86.89 | 86.34 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 84.51 | 86.07 | 86.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 84.26 | 85.48 | 85.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 83.61 | 82.82 | 83.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 83.61 | 82.82 | 83.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 83.61 | 82.82 | 83.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 83.80 | 82.82 | 83.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 84.52 | 83.16 | 83.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 84.52 | 83.16 | 83.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 83.79 | 83.28 | 83.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 83.48 | 83.28 | 83.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 84.85 | 83.87 | 84.00 | SL hit (close>static) qty=1.00 sl=84.59 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 86.36 | 84.53 | 84.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 87.04 | 85.55 | 84.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 13:15:00 | 91.80 | 92.00 | 90.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 13:30:00 | 92.07 | 92.00 | 90.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 91.00 | 91.60 | 90.77 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 90.72 | 91.58 | 91.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 89.90 | 91.10 | 91.36 | Break + close below crossover candle low |

### Cycle 115 — BUY (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 09:15:00 | 94.37 | 91.49 | 91.45 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 91.30 | 92.55 | 92.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 90.82 | 92.02 | 92.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 89.05 | 88.94 | 90.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 89.05 | 88.94 | 90.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 90.60 | 89.26 | 90.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:45:00 | 90.25 | 89.26 | 90.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 90.68 | 89.55 | 90.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:45:00 | 91.40 | 89.55 | 90.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 89.79 | 89.67 | 90.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 89.11 | 89.67 | 90.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 90.76 | 90.00 | 90.18 | SL hit (close>static) qty=1.00 sl=90.39 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 96.33 | 89.76 | 89.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 100.82 | 96.63 | 93.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 102.50 | 103.31 | 100.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 102.50 | 103.31 | 100.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 100.77 | 102.46 | 100.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 101.14 | 102.46 | 100.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 103.54 | 102.67 | 100.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 106.70 | 103.98 | 102.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-07 09:15:00 | 117.37 | 110.25 | 107.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 107.77 | 110.11 | 110.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 105.50 | 107.57 | 108.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 106.67 | 106.51 | 107.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 106.67 | 106.51 | 107.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 106.85 | 106.58 | 107.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:45:00 | 107.33 | 106.58 | 107.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 107.50 | 106.72 | 107.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:15:00 | 106.93 | 106.72 | 107.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 108.35 | 107.04 | 107.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:00:00 | 106.16 | 106.87 | 107.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:45:00 | 105.91 | 106.69 | 107.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:30:00 | 105.79 | 106.70 | 107.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 105.85 | 106.35 | 106.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 106.76 | 105.97 | 106.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 106.76 | 105.97 | 106.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 107.47 | 106.27 | 106.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:00:00 | 107.47 | 106.27 | 106.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 108.22 | 106.66 | 106.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:45:00 | 108.25 | 106.66 | 106.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 108.18 | 106.96 | 106.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 108.18 | 106.96 | 106.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 109.10 | 107.47 | 107.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 11:15:00 | 106.89 | 107.61 | 107.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 11:15:00 | 106.89 | 107.61 | 107.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 106.89 | 107.61 | 107.34 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 104.89 | 106.77 | 107.00 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 12:15:00 | 107.61 | 107.03 | 107.01 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 106.04 | 106.83 | 106.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 12:15:00 | 105.52 | 106.38 | 106.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 107.86 | 106.09 | 106.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 107.86 | 106.09 | 106.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 107.86 | 106.09 | 106.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 108.78 | 106.09 | 106.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 109.20 | 106.71 | 106.64 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 13:15:00 | 105.27 | 106.48 | 106.57 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 09:15:00 | 107.59 | 106.63 | 106.61 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 106.00 | 106.50 | 106.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 104.83 | 106.08 | 106.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 105.93 | 105.51 | 105.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 105.93 | 105.51 | 105.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 105.93 | 105.51 | 105.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:45:00 | 106.32 | 105.51 | 105.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 105.75 | 105.56 | 105.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:15:00 | 105.09 | 105.56 | 105.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 105.00 | 105.45 | 105.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 13:45:00 | 104.88 | 105.25 | 105.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 106.28 | 105.38 | 105.62 | SL hit (close>static) qty=1.00 sl=106.08 alert=retest2 |

### Cycle 127 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 107.80 | 105.10 | 105.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 110.70 | 106.22 | 105.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 12:15:00 | 107.79 | 108.39 | 107.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 13:00:00 | 107.79 | 108.39 | 107.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 107.82 | 108.21 | 107.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:45:00 | 107.24 | 108.21 | 107.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 107.79 | 108.36 | 107.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:45:00 | 107.45 | 108.36 | 107.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 107.68 | 108.22 | 107.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:00:00 | 107.68 | 108.22 | 107.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 107.56 | 108.09 | 107.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:30:00 | 107.28 | 108.09 | 107.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 107.40 | 107.95 | 107.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 108.13 | 107.95 | 107.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:00:00 | 107.90 | 108.08 | 107.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 109.56 | 107.79 | 107.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 14:15:00 | 109.64 | 111.05 | 111.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 109.64 | 111.05 | 111.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 108.95 | 110.27 | 110.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 110.26 | 109.90 | 110.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 110.26 | 109.90 | 110.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 109.72 | 109.86 | 110.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 110.40 | 109.86 | 110.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 109.89 | 109.87 | 110.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 109.67 | 109.85 | 110.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 110.46 | 108.41 | 108.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 110.46 | 108.41 | 108.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 110.50 | 108.83 | 108.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 112.80 | 112.87 | 111.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:30:00 | 112.34 | 112.87 | 111.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 110.92 | 112.47 | 111.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:30:00 | 110.87 | 112.47 | 111.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 111.54 | 112.28 | 111.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:45:00 | 110.98 | 112.28 | 111.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 111.80 | 112.09 | 111.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:15:00 | 109.88 | 112.09 | 111.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 110.10 | 111.70 | 111.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 111.55 | 111.70 | 111.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 110.90 | 111.64 | 111.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 10:15:00 | 110.90 | 111.64 | 111.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 108.99 | 110.97 | 111.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 13:15:00 | 108.03 | 107.64 | 109.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 14:00:00 | 108.03 | 107.64 | 109.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 108.90 | 107.90 | 109.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:30:00 | 108.83 | 107.90 | 109.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 109.43 | 108.34 | 109.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:45:00 | 110.23 | 108.34 | 109.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 109.31 | 108.53 | 109.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:00:00 | 109.31 | 108.53 | 109.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 110.46 | 108.92 | 109.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 110.46 | 108.92 | 109.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 110.65 | 109.64 | 109.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 11:15:00 | 111.00 | 110.28 | 109.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 10:15:00 | 111.65 | 111.99 | 111.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 11:00:00 | 111.65 | 111.99 | 111.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 109.89 | 111.57 | 111.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 109.89 | 111.57 | 111.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 110.18 | 111.29 | 111.14 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 109.38 | 110.91 | 110.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 109.10 | 110.55 | 110.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 110.48 | 109.52 | 110.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 110.48 | 109.52 | 110.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 110.48 | 109.52 | 110.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 110.48 | 109.52 | 110.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 110.05 | 109.63 | 110.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 110.32 | 109.63 | 110.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 110.70 | 109.84 | 110.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 111.46 | 109.84 | 110.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 112.21 | 110.31 | 110.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 113.85 | 111.34 | 110.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 14:15:00 | 114.25 | 114.76 | 113.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 15:00:00 | 114.25 | 114.76 | 113.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 119.60 | 120.82 | 119.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 122.08 | 120.82 | 119.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 10:00:00 | 120.98 | 122.86 | 121.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 121.02 | 122.49 | 121.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 121.25 | 122.01 | 121.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 121.25 | 121.86 | 121.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 121.25 | 121.86 | 121.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 121.30 | 121.75 | 121.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:15:00 | 121.00 | 121.75 | 121.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 121.00 | 121.60 | 121.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 122.85 | 121.60 | 121.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 121.19 | 121.52 | 121.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 119.89 | 121.19 | 121.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 119.89 | 121.19 | 121.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 118.82 | 120.72 | 121.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 106.75 | 106.23 | 109.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:45:00 | 105.86 | 106.23 | 109.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 106.44 | 106.52 | 108.24 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 110.23 | 109.04 | 108.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 115.84 | 110.40 | 109.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 13:15:00 | 115.57 | 115.71 | 113.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 14:00:00 | 115.57 | 115.71 | 113.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 114.60 | 115.72 | 114.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:30:00 | 114.43 | 115.72 | 114.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 114.47 | 115.47 | 114.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:30:00 | 114.48 | 115.47 | 114.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 114.72 | 115.32 | 114.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 118.22 | 115.07 | 114.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 116.08 | 115.50 | 114.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-22 10:15:00 | 130.04 | 121.61 | 118.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 118.75 | 121.26 | 121.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 112.68 | 119.15 | 120.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 112.80 | 111.69 | 114.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 112.80 | 111.69 | 114.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 111.65 | 111.74 | 114.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 113.74 | 111.74 | 114.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 114.00 | 112.39 | 114.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 114.91 | 112.39 | 114.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 114.00 | 112.71 | 114.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:45:00 | 113.68 | 113.14 | 113.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 15:15:00 | 114.80 | 113.80 | 114.11 | SL hit (close>static) qty=1.00 sl=114.67 alert=retest2 |

### Cycle 137 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 117.59 | 114.56 | 114.42 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 112.39 | 114.12 | 114.33 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 117.59 | 114.10 | 114.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 10:15:00 | 119.18 | 115.12 | 114.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 115.69 | 116.02 | 115.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 11:15:00 | 115.69 | 116.02 | 115.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 115.69 | 116.02 | 115.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 115.69 | 116.02 | 115.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 115.65 | 115.94 | 115.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:45:00 | 115.20 | 115.94 | 115.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 115.38 | 115.83 | 115.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:15:00 | 116.00 | 115.83 | 115.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 116.55 | 115.97 | 115.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 15:15:00 | 119.10 | 115.97 | 115.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 113.53 | 115.99 | 115.65 | SL hit (close<static) qty=1.00 sl=114.81 alert=retest2 |

### Cycle 140 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 112.78 | 115.34 | 115.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 11:15:00 | 111.07 | 114.49 | 115.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 100.70 | 98.23 | 99.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 100.70 | 98.23 | 99.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 100.70 | 98.23 | 99.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 100.08 | 98.23 | 99.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 100.94 | 98.77 | 100.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 100.15 | 98.97 | 100.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 99.75 | 98.97 | 100.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 97.82 | 99.12 | 99.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 102.03 | 98.90 | 99.21 | SL hit (close>static) qty=1.00 sl=101.77 alert=retest2 |

### Cycle 141 — BUY (started 2025-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 11:15:00 | 99.55 | 99.41 | 99.41 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 98.13 | 99.20 | 99.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 11:15:00 | 97.23 | 98.61 | 99.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 87.97 | 87.92 | 89.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 11:45:00 | 88.19 | 87.92 | 89.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 89.16 | 88.05 | 88.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 89.81 | 88.05 | 88.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 88.30 | 88.10 | 88.91 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 13:15:00 | 94.22 | 90.34 | 89.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 14:15:00 | 96.20 | 91.51 | 90.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 10:15:00 | 91.66 | 92.25 | 91.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-27 11:00:00 | 91.66 | 92.25 | 91.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 91.18 | 92.04 | 91.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:00:00 | 91.18 | 92.04 | 91.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 90.97 | 91.83 | 91.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:30:00 | 91.05 | 91.83 | 91.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 91.00 | 91.66 | 91.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:30:00 | 90.67 | 91.66 | 91.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 90.75 | 91.48 | 91.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 90.75 | 91.48 | 91.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 89.95 | 91.17 | 90.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:15:00 | 89.91 | 91.17 | 90.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 88.00 | 90.54 | 90.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 85.29 | 87.58 | 88.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 86.72 | 86.41 | 87.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 86.72 | 86.41 | 87.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 88.22 | 86.77 | 87.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 88.05 | 86.77 | 87.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 90.11 | 87.44 | 88.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:30:00 | 90.18 | 87.44 | 88.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 90.04 | 87.96 | 88.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 89.33 | 87.96 | 88.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 93.10 | 88.99 | 88.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 93.10 | 88.99 | 88.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 93.53 | 89.90 | 89.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 95.50 | 95.63 | 93.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 95.50 | 95.63 | 93.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 94.02 | 95.23 | 94.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:00:00 | 94.02 | 95.23 | 94.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 93.06 | 94.80 | 93.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 93.06 | 94.80 | 93.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 93.38 | 94.51 | 93.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 94.54 | 94.51 | 93.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 92.76 | 94.20 | 94.16 | SL hit (close<static) qty=1.00 sl=92.80 alert=retest2 |

### Cycle 146 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 92.61 | 93.88 | 94.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 92.45 | 93.42 | 93.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 91.97 | 91.64 | 92.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 91.97 | 91.64 | 92.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 91.97 | 91.64 | 92.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 91.34 | 91.65 | 92.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 91.50 | 91.66 | 92.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 14:15:00 | 92.92 | 92.14 | 92.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 92.92 | 92.14 | 92.10 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 91.49 | 92.01 | 92.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 90.97 | 91.72 | 91.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 91.04 | 90.57 | 91.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 10:15:00 | 91.04 | 90.57 | 91.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 91.04 | 90.57 | 91.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:30:00 | 90.99 | 90.57 | 91.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 91.07 | 90.67 | 91.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:30:00 | 92.08 | 90.67 | 91.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 90.50 | 90.63 | 90.96 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 12:15:00 | 91.22 | 91.06 | 91.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 94.36 | 91.79 | 91.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 101.00 | 101.41 | 99.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 14:45:00 | 101.15 | 101.41 | 99.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 101.55 | 102.68 | 101.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 101.55 | 102.68 | 101.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 101.08 | 102.36 | 101.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 102.21 | 101.84 | 101.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-02 09:15:00 | 112.43 | 108.65 | 106.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 140.00 | 141.70 | 141.73 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 148.50 | 143.06 | 142.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 149.74 | 145.41 | 143.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 147.45 | 148.55 | 146.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 12:45:00 | 147.08 | 148.55 | 146.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 146.53 | 148.14 | 146.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:45:00 | 146.10 | 148.14 | 146.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 145.38 | 147.59 | 146.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 145.38 | 147.59 | 146.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 145.50 | 147.17 | 146.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 144.99 | 147.17 | 146.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 145.99 | 146.91 | 146.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 145.99 | 146.91 | 146.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 144.70 | 146.47 | 146.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:00:00 | 144.70 | 146.47 | 146.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 145.43 | 146.12 | 146.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 144.28 | 145.75 | 145.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 12:15:00 | 145.01 | 144.98 | 145.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:00:00 | 145.01 | 144.98 | 145.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 144.99 | 144.98 | 145.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:30:00 | 144.40 | 145.18 | 145.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 146.28 | 145.40 | 145.49 | SL hit (close>static) qty=1.00 sl=145.70 alert=retest2 |

### Cycle 153 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 146.84 | 145.54 | 145.50 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 144.24 | 145.51 | 145.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 142.62 | 144.73 | 145.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 141.38 | 140.83 | 142.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 141.38 | 140.83 | 142.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 142.40 | 141.01 | 142.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:45:00 | 141.39 | 141.01 | 142.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 144.10 | 141.63 | 142.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 143.71 | 141.63 | 142.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 144.16 | 142.14 | 142.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:45:00 | 141.49 | 142.21 | 142.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 134.42 | 139.77 | 141.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 140.57 | 139.68 | 140.83 | SL hit (close>ema200) qty=0.50 sl=139.68 alert=retest2 |

### Cycle 155 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 147.85 | 142.04 | 141.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 150.84 | 143.80 | 142.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 147.89 | 148.14 | 146.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 147.89 | 148.14 | 146.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 145.66 | 153.44 | 152.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:00:00 | 145.66 | 153.44 | 152.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 144.88 | 151.73 | 152.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 15:15:00 | 144.71 | 148.48 | 150.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 11:15:00 | 148.85 | 148.00 | 149.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 148.85 | 148.00 | 149.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 149.63 | 148.32 | 149.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 149.63 | 148.32 | 149.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 149.60 | 148.58 | 149.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:15:00 | 150.94 | 148.58 | 149.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 151.60 | 149.18 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 151.60 | 149.18 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 151.00 | 149.55 | 149.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 149.90 | 149.55 | 149.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 154.60 | 150.56 | 150.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 163.05 | 154.03 | 152.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 14:15:00 | 155.95 | 155.97 | 154.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:45:00 | 155.66 | 155.97 | 154.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 155.90 | 155.96 | 154.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 153.34 | 155.85 | 154.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 154.45 | 155.51 | 154.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 154.45 | 155.51 | 154.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 154.19 | 155.25 | 154.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 154.18 | 155.25 | 154.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 154.99 | 155.20 | 154.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:45:00 | 155.99 | 155.54 | 154.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 155.45 | 155.95 | 155.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 171.59 | 166.41 | 163.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 173.00 | 173.98 | 174.03 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 174.82 | 174.15 | 174.10 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 172.50 | 174.47 | 174.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 171.82 | 173.77 | 174.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 13:15:00 | 172.60 | 172.50 | 173.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 13:45:00 | 173.01 | 172.50 | 173.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 172.58 | 172.51 | 173.21 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 175.71 | 173.62 | 173.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 176.85 | 174.56 | 174.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 176.23 | 176.50 | 175.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 176.23 | 176.50 | 175.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 174.46 | 176.09 | 175.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 174.50 | 176.09 | 175.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 175.66 | 176.01 | 175.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 177.70 | 176.16 | 175.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 14:15:00 | 173.26 | 175.70 | 175.68 | SL hit (close<static) qty=1.00 sl=174.17 alert=retest2 |

### Cycle 162 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 173.55 | 175.27 | 175.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 172.50 | 174.09 | 174.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 171.31 | 171.25 | 172.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 171.31 | 171.25 | 172.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 173.23 | 171.68 | 172.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 173.94 | 171.68 | 172.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 171.70 | 171.69 | 172.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 171.24 | 171.69 | 172.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 162.68 | 165.33 | 166.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 165.80 | 165.42 | 166.80 | SL hit (close>ema200) qty=0.50 sl=165.42 alert=retest2 |

### Cycle 163 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 165.99 | 165.49 | 165.44 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 162.39 | 164.91 | 165.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 161.29 | 164.18 | 164.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 161.50 | 159.57 | 161.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 161.50 | 159.57 | 161.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 161.50 | 159.57 | 161.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 161.25 | 159.57 | 161.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 161.02 | 159.86 | 161.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 160.00 | 160.34 | 161.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 164.64 | 160.99 | 161.11 | SL hit (close>static) qty=1.00 sl=161.79 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 162.80 | 161.35 | 161.26 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 160.07 | 161.12 | 161.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 158.74 | 160.64 | 160.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 159.67 | 159.62 | 160.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:45:00 | 159.75 | 159.62 | 160.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 160.49 | 159.43 | 159.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 160.80 | 159.43 | 159.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 160.31 | 159.61 | 159.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 160.36 | 159.61 | 159.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 161.05 | 160.16 | 160.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 162.70 | 160.67 | 160.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 161.14 | 161.27 | 160.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 161.14 | 161.27 | 160.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 160.59 | 161.12 | 160.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 160.59 | 161.12 | 160.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 159.90 | 160.88 | 160.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 161.61 | 160.88 | 160.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:45:00 | 160.70 | 160.84 | 160.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 158.92 | 160.45 | 160.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 158.92 | 160.45 | 160.59 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 161.05 | 159.97 | 159.97 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 159.32 | 159.84 | 159.91 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 160.10 | 159.94 | 159.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 160.90 | 160.13 | 160.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 160.61 | 161.48 | 161.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 160.61 | 161.48 | 161.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 160.61 | 161.48 | 161.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 160.61 | 161.48 | 161.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 161.20 | 161.42 | 161.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 161.70 | 161.50 | 161.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-15 10:15:00 | 177.87 | 170.41 | 166.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 190.45 | 191.00 | 191.06 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 200.10 | 192.82 | 191.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 217.74 | 201.10 | 196.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 210.15 | 213.32 | 206.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:45:00 | 210.72 | 213.32 | 206.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 206.80 | 211.27 | 206.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 207.39 | 211.27 | 206.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 208.00 | 210.61 | 207.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 208.00 | 210.61 | 207.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 209.20 | 209.89 | 207.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:15:00 | 208.00 | 209.89 | 207.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 208.00 | 209.52 | 207.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 211.49 | 209.52 | 207.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 217.68 | 211.15 | 208.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:30:00 | 217.99 | 216.35 | 213.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 218.31 | 216.60 | 213.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 220.61 | 216.97 | 214.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 225.87 | 226.35 | 226.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 225.87 | 226.35 | 226.35 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 227.20 | 226.46 | 226.40 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 219.89 | 225.54 | 226.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 216.63 | 221.03 | 223.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 204.59 | 203.93 | 208.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 12:00:00 | 199.20 | 202.98 | 207.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 207.33 | 202.37 | 205.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 207.33 | 202.37 | 205.15 | SL hit (close>ema400) qty=1.00 sl=205.15 alert=retest1 |

### Cycle 177 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 208.84 | 206.51 | 206.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 13:15:00 | 212.10 | 208.93 | 207.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 13:15:00 | 219.79 | 219.99 | 216.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:30:00 | 219.86 | 219.99 | 216.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 224.18 | 224.13 | 221.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:45:00 | 225.70 | 224.51 | 221.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:30:00 | 227.27 | 226.03 | 223.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:30:00 | 226.80 | 226.06 | 224.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 218.06 | 223.41 | 223.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 218.06 | 223.41 | 223.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 217.52 | 221.09 | 222.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 225.43 | 220.64 | 221.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 225.43 | 220.64 | 221.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 225.43 | 220.64 | 221.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 225.43 | 220.64 | 221.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 224.75 | 221.46 | 221.95 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 225.80 | 222.33 | 222.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 226.60 | 223.19 | 222.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 224.18 | 225.25 | 224.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 224.18 | 225.25 | 224.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 224.18 | 225.25 | 224.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 224.18 | 225.25 | 224.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 225.66 | 225.33 | 224.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:15:00 | 223.83 | 225.33 | 224.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 218.33 | 223.93 | 223.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:45:00 | 217.65 | 223.93 | 223.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 214.37 | 222.02 | 222.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 15:15:00 | 211.10 | 213.21 | 216.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 185.75 | 185.61 | 191.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 14:00:00 | 185.75 | 185.61 | 191.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 185.29 | 183.39 | 186.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 181.07 | 183.15 | 185.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:15:00 | 172.02 | 178.64 | 182.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 169.34 | 168.59 | 171.93 | SL hit (close>ema200) qty=0.50 sl=168.59 alert=retest2 |

### Cycle 181 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 173.14 | 171.80 | 171.75 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 169.55 | 171.42 | 171.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 168.95 | 170.58 | 171.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 11:15:00 | 172.90 | 170.34 | 170.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 11:15:00 | 172.90 | 170.34 | 170.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 172.90 | 170.34 | 170.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 172.90 | 170.34 | 170.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 174.90 | 171.25 | 171.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 179.43 | 173.52 | 172.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 176.62 | 177.78 | 176.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 176.62 | 177.78 | 176.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 176.62 | 177.78 | 176.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 176.69 | 177.78 | 176.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 174.67 | 177.16 | 175.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 174.67 | 177.16 | 175.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 175.36 | 176.80 | 175.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 175.76 | 176.62 | 175.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 173.90 | 176.08 | 175.66 | SL hit (close<static) qty=1.00 sl=174.25 alert=retest2 |

### Cycle 184 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 173.99 | 175.27 | 175.34 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 177.51 | 175.72 | 175.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 11:15:00 | 180.23 | 176.62 | 175.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 183.27 | 189.34 | 185.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 183.27 | 189.34 | 185.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 183.27 | 189.34 | 185.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 183.27 | 189.34 | 185.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 184.29 | 188.33 | 185.70 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 182.00 | 184.62 | 184.66 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 193.30 | 186.35 | 185.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 197.60 | 193.18 | 189.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 11:15:00 | 192.70 | 193.25 | 190.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 12:00:00 | 192.70 | 193.25 | 190.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 190.60 | 194.55 | 193.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 190.60 | 194.55 | 193.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 193.29 | 194.29 | 193.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:45:00 | 193.61 | 193.54 | 193.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 193.65 | 193.54 | 193.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 189.65 | 194.79 | 194.65 | SL hit (close<static) qty=1.00 sl=190.52 alert=retest2 |

### Cycle 188 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 190.16 | 193.86 | 194.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 188.80 | 192.85 | 193.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 11:15:00 | 178.51 | 178.37 | 181.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:30:00 | 179.62 | 178.37 | 181.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 175.75 | 174.55 | 176.22 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 178.26 | 177.11 | 177.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 179.31 | 177.96 | 177.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 176.31 | 177.78 | 177.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 176.31 | 177.78 | 177.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 176.31 | 177.78 | 177.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 176.31 | 177.78 | 177.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 176.33 | 177.49 | 177.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:15:00 | 177.39 | 177.41 | 177.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 175.80 | 177.08 | 177.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 175.80 | 177.08 | 177.21 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 177.95 | 177.29 | 177.28 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 176.85 | 177.23 | 177.28 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 178.15 | 177.31 | 177.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 179.12 | 177.67 | 177.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 176.39 | 177.49 | 177.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 176.39 | 177.49 | 177.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 176.39 | 177.49 | 177.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 176.39 | 177.49 | 177.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 176.14 | 177.22 | 177.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 175.20 | 176.81 | 177.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 172.99 | 172.94 | 174.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 11:00:00 | 172.99 | 172.94 | 174.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 172.65 | 172.88 | 174.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 171.80 | 172.75 | 173.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 171.43 | 170.32 | 171.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 170.70 | 170.43 | 170.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 13:15:00 | 163.21 | 166.52 | 167.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 13:15:00 | 162.86 | 166.52 | 167.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 13:15:00 | 162.16 | 166.52 | 167.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 168.61 | 166.32 | 167.06 | SL hit (close>ema200) qty=0.50 sl=166.32 alert=retest2 |

### Cycle 195 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 170.40 | 167.82 | 167.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 14:15:00 | 174.39 | 169.13 | 168.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 10:15:00 | 168.44 | 169.67 | 168.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 168.44 | 169.67 | 168.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 168.44 | 169.67 | 168.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 168.44 | 169.67 | 168.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 168.55 | 169.44 | 168.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:15:00 | 168.72 | 169.44 | 168.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 168.80 | 169.31 | 168.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 168.80 | 169.31 | 168.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 168.45 | 169.14 | 168.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 168.45 | 169.14 | 168.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 167.18 | 168.75 | 168.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 167.18 | 168.75 | 168.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 167.24 | 168.45 | 168.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 159.65 | 166.69 | 167.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 161.33 | 161.19 | 163.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 161.33 | 161.19 | 163.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 161.33 | 161.19 | 163.71 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 165.05 | 163.11 | 163.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 10:15:00 | 166.52 | 163.79 | 163.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 163.00 | 164.98 | 164.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 163.00 | 164.98 | 164.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 163.00 | 164.98 | 164.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 162.74 | 164.98 | 164.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 162.86 | 164.56 | 164.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 162.89 | 164.56 | 164.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 162.97 | 163.83 | 163.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 162.15 | 163.49 | 163.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 156.72 | 153.82 | 155.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 156.72 | 153.82 | 155.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 156.72 | 153.82 | 155.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 156.72 | 153.82 | 155.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 158.15 | 154.69 | 156.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 157.68 | 154.69 | 156.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 158.44 | 156.53 | 156.69 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 158.39 | 156.90 | 156.84 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 154.83 | 156.72 | 156.85 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 158.80 | 156.45 | 156.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 162.13 | 158.83 | 157.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 12:15:00 | 159.08 | 159.23 | 158.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:00:00 | 159.08 | 159.23 | 158.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 164.29 | 160.58 | 159.21 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 153.30 | 158.70 | 159.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 152.71 | 154.18 | 155.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 154.11 | 152.92 | 153.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 154.11 | 152.92 | 153.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 154.11 | 152.92 | 153.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 154.11 | 152.92 | 153.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 154.25 | 153.19 | 153.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 153.22 | 153.19 | 153.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 155.61 | 153.67 | 154.01 | SL hit (close>static) qty=1.00 sl=154.70 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 157.50 | 154.63 | 154.40 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 153.74 | 154.99 | 155.08 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 156.10 | 155.28 | 155.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 156.90 | 155.60 | 155.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 155.58 | 155.68 | 155.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 11:45:00 | 155.76 | 155.68 | 155.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 153.97 | 155.53 | 155.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 154.10 | 155.53 | 155.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 153.80 | 155.19 | 155.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 153.26 | 154.80 | 155.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 13:15:00 | 154.50 | 154.44 | 154.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 154.50 | 154.44 | 154.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 154.53 | 154.46 | 154.87 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 163.16 | 156.33 | 155.65 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 155.19 | 157.06 | 157.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 155.00 | 156.13 | 156.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 153.44 | 153.25 | 154.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:00:00 | 153.44 | 153.25 | 154.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 155.35 | 153.77 | 154.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 155.35 | 153.77 | 154.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 155.50 | 154.12 | 154.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 159.26 | 154.12 | 154.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 161.55 | 155.61 | 155.12 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 159.01 | 159.81 | 159.83 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 161.55 | 160.08 | 159.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 168.28 | 162.97 | 161.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 12:15:00 | 164.20 | 165.21 | 164.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 12:15:00 | 164.20 | 165.21 | 164.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 164.20 | 165.21 | 164.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 163.85 | 165.21 | 164.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 163.65 | 164.89 | 164.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 163.65 | 164.89 | 164.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 164.16 | 164.75 | 164.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 165.10 | 164.75 | 164.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:45:00 | 165.00 | 164.73 | 164.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 164.95 | 164.80 | 164.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 165.26 | 164.82 | 164.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 164.29 | 164.71 | 164.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 164.60 | 164.71 | 164.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 164.85 | 164.74 | 164.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 161.11 | 162.85 | 163.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 158.40 | 157.90 | 159.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 158.40 | 157.90 | 159.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 155.83 | 157.82 | 158.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 155.40 | 157.82 | 158.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 147.63 | 150.27 | 153.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 146.45 | 146.03 | 148.85 | SL hit (close>ema200) qty=0.50 sl=146.03 alert=retest2 |

### Cycle 213 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 133.28 | 130.82 | 130.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 137.83 | 133.25 | 132.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 138.90 | 139.23 | 136.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 138.90 | 139.23 | 136.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 137.00 | 138.78 | 136.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 137.07 | 138.78 | 136.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 135.64 | 138.15 | 136.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 135.64 | 138.15 | 136.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 135.71 | 137.66 | 136.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 135.47 | 137.66 | 136.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 135.45 | 137.22 | 136.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 130.51 | 137.22 | 136.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 130.70 | 135.92 | 135.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 128.50 | 134.43 | 135.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 132.69 | 132.07 | 133.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 132.69 | 132.07 | 133.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 132.69 | 132.07 | 133.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 127.66 | 132.07 | 133.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 128.61 | 131.37 | 133.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 14:00:00 | 124.85 | 127.74 | 129.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 124.85 | 127.28 | 129.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 129.49 | 126.24 | 126.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 129.49 | 126.24 | 126.17 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 124.67 | 126.17 | 126.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 122.29 | 125.40 | 125.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 15:15:00 | 123.23 | 123.22 | 124.39 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 120.37 | 122.79 | 124.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 123.92 | 123.09 | 124.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 123.92 | 123.09 | 124.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 122.68 | 123.01 | 123.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:45:00 | 122.30 | 122.90 | 123.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:00:00 | 122.06 | 122.73 | 123.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 124.78 | 123.14 | 123.56 | SL hit (close>ema400) qty=1.00 sl=123.56 alert=retest1 |

### Cycle 217 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 124.30 | 123.30 | 123.17 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 120.87 | 122.86 | 123.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 120.20 | 121.63 | 122.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 118.41 | 118.37 | 119.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:30:00 | 118.30 | 118.37 | 119.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 117.13 | 118.08 | 119.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 115.18 | 117.05 | 118.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:45:00 | 115.58 | 116.52 | 117.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 116.36 | 115.58 | 116.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 117.01 | 116.60 | 116.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 117.01 | 116.60 | 116.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 118.74 | 117.03 | 116.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 117.30 | 118.10 | 117.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 117.30 | 118.10 | 117.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 117.30 | 118.10 | 117.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 118.75 | 118.10 | 117.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 116.78 | 117.83 | 117.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 116.78 | 117.83 | 117.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 118.05 | 117.88 | 117.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:45:00 | 116.60 | 117.88 | 117.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 118.31 | 119.32 | 118.47 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 114.63 | 117.65 | 117.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 14:15:00 | 114.29 | 116.55 | 117.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 109.73 | 109.36 | 111.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 109.73 | 109.36 | 111.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 104.85 | 103.71 | 105.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 107.04 | 103.71 | 105.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 108.12 | 104.59 | 105.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 108.12 | 104.59 | 105.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 107.34 | 105.14 | 106.06 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 108.45 | 106.84 | 106.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 110.16 | 107.50 | 106.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 111.44 | 111.63 | 109.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:30:00 | 111.00 | 111.63 | 109.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 109.66 | 111.95 | 111.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 109.66 | 111.95 | 111.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 108.57 | 111.28 | 110.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 108.57 | 111.28 | 110.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 110.11 | 110.76 | 110.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 108.77 | 110.76 | 110.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 107.35 | 110.08 | 110.35 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 113.11 | 110.20 | 110.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 113.90 | 112.57 | 111.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 113.00 | 114.12 | 112.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 113.00 | 114.12 | 112.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 113.00 | 114.12 | 112.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 112.29 | 114.12 | 112.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 112.15 | 113.73 | 112.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 112.15 | 113.73 | 112.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 113.06 | 113.59 | 112.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:30:00 | 113.25 | 113.67 | 112.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:30:00 | 113.60 | 113.46 | 112.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 114.20 | 113.46 | 112.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 113.18 | 113.64 | 113.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 113.85 | 113.68 | 113.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 108.35 | 111.25 | 112.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 108.83 | 108.26 | 109.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 108.83 | 108.26 | 109.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 108.83 | 108.26 | 109.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 109.79 | 108.26 | 109.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 111.25 | 109.01 | 109.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 110.91 | 109.01 | 109.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 111.27 | 109.46 | 110.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:15:00 | 111.10 | 109.46 | 110.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 111.00 | 109.87 | 110.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 110.75 | 109.87 | 110.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 111.16 | 110.13 | 110.21 | EMA400 retest candle locked (from downside) |

### Cycle 225 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 111.40 | 110.38 | 110.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 112.08 | 110.95 | 110.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 110.07 | 110.98 | 110.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 110.07 | 110.98 | 110.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 110.07 | 110.98 | 110.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 110.07 | 110.98 | 110.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 108.88 | 110.56 | 110.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 108.95 | 110.56 | 110.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 109.27 | 110.30 | 110.43 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 111.61 | 110.47 | 110.47 | EMA200 above EMA400 |

### Cycle 228 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 108.35 | 110.20 | 110.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 107.35 | 109.18 | 109.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 110.79 | 109.18 | 109.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 110.79 | 109.18 | 109.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 110.79 | 109.18 | 109.63 | EMA400 retest candle locked (from downside) |

### Cycle 229 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 115.75 | 110.78 | 110.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 116.77 | 112.75 | 111.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 110.65 | 113.20 | 111.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 110.65 | 113.20 | 111.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 110.65 | 113.20 | 111.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 110.65 | 113.20 | 111.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 110.66 | 112.69 | 111.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 111.10 | 112.69 | 111.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 11:15:00 | 122.21 | 117.92 | 116.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 230 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 134.30 | 134.82 | 134.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 132.29 | 134.31 | 134.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 133.75 | 133.45 | 134.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 133.75 | 133.45 | 134.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 128.46 | 129.31 | 130.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 128.30 | 129.31 | 130.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 128.29 | 129.00 | 130.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 121.89 | 125.73 | 128.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 121.88 | 125.73 | 128.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 123.86 | 123.72 | 125.78 | SL hit (close>ema200) qty=0.50 sl=123.72 alert=retest2 |

### Cycle 231 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 127.65 | 125.42 | 125.29 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 124.78 | 125.46 | 125.49 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 12:15:00 | 71.75 | 2024-04-15 09:15:00 | 68.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 13:15:00 | 71.70 | 2024-04-15 09:15:00 | 68.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 12:15:00 | 71.75 | 2024-04-16 15:15:00 | 70.00 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2024-04-12 13:15:00 | 71.70 | 2024-04-16 15:15:00 | 70.00 | STOP_HIT | 0.50 | 2.37% |
| BUY | retest2 | 2024-04-26 09:15:00 | 71.60 | 2024-04-26 09:15:00 | 71.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-04-26 11:00:00 | 72.00 | 2024-04-26 12:15:00 | 70.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-05-09 10:15:00 | 66.60 | 2024-05-14 13:15:00 | 66.50 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-05-18 09:15:00 | 70.45 | 2024-05-23 10:15:00 | 69.15 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-05-21 09:45:00 | 69.70 | 2024-05-23 10:15:00 | 69.15 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-21 12:45:00 | 69.70 | 2024-05-23 10:15:00 | 69.15 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-21 13:30:00 | 70.70 | 2024-05-23 10:15:00 | 69.15 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-05-28 12:15:00 | 68.20 | 2024-05-29 13:15:00 | 71.10 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-05-28 13:30:00 | 68.25 | 2024-05-29 13:15:00 | 71.10 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2024-07-15 13:45:00 | 88.08 | 2024-07-15 14:15:00 | 88.33 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-07-18 13:15:00 | 90.75 | 2024-07-19 09:15:00 | 88.01 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-07-22 12:15:00 | 88.27 | 2024-07-23 11:15:00 | 83.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:00:00 | 88.29 | 2024-07-23 11:15:00 | 83.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:30:00 | 88.21 | 2024-07-23 12:15:00 | 83.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 88.27 | 2024-07-24 11:15:00 | 85.21 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2024-07-22 14:00:00 | 88.29 | 2024-07-24 11:15:00 | 85.21 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2024-07-22 14:30:00 | 88.21 | 2024-07-24 11:15:00 | 85.21 | STOP_HIT | 0.50 | 3.40% |
| BUY | retest2 | 2024-08-21 09:15:00 | 87.75 | 2024-08-26 09:15:00 | 86.48 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-08-27 14:45:00 | 87.73 | 2024-08-28 10:15:00 | 87.92 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-08-28 10:00:00 | 87.95 | 2024-08-28 10:15:00 | 87.92 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-08-30 11:00:00 | 85.50 | 2024-09-05 11:15:00 | 85.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-08-30 11:30:00 | 85.50 | 2024-09-05 11:15:00 | 85.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-09-05 11:15:00 | 85.38 | 2024-09-05 11:15:00 | 85.40 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-09-06 11:15:00 | 86.10 | 2024-09-11 15:15:00 | 85.34 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-09-06 14:30:00 | 86.51 | 2024-09-11 15:15:00 | 85.34 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-09-09 10:45:00 | 86.20 | 2024-09-11 15:15:00 | 85.34 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-10 09:15:00 | 87.75 | 2024-09-11 15:15:00 | 85.34 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-09-27 11:45:00 | 84.25 | 2024-10-01 11:15:00 | 86.98 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-10-01 10:30:00 | 84.11 | 2024-10-01 11:15:00 | 86.98 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-10-08 12:15:00 | 83.48 | 2024-10-08 14:15:00 | 84.85 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-10-24 09:15:00 | 89.11 | 2024-10-24 11:15:00 | 90.76 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-10-25 09:15:00 | 88.60 | 2024-10-29 09:15:00 | 96.33 | STOP_HIT | 1.00 | -8.72% |
| BUY | retest2 | 2024-11-05 09:30:00 | 106.70 | 2024-11-07 09:15:00 | 117.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-14 11:00:00 | 106.16 | 2024-11-18 12:15:00 | 108.18 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-14 11:45:00 | 105.91 | 2024-11-18 12:15:00 | 108.18 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-11-14 12:30:00 | 105.79 | 2024-11-18 12:15:00 | 108.18 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-11-14 13:30:00 | 105.85 | 2024-11-18 12:15:00 | 108.18 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-11-27 13:45:00 | 104.88 | 2024-11-28 09:15:00 | 106.28 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-11-28 12:00:00 | 104.09 | 2024-11-29 13:15:00 | 107.80 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-11-29 11:45:00 | 104.00 | 2024-11-29 13:15:00 | 107.80 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2024-12-04 09:15:00 | 108.13 | 2024-12-09 14:15:00 | 109.64 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2024-12-04 13:00:00 | 107.90 | 2024-12-09 14:15:00 | 109.64 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2024-12-05 09:15:00 | 109.56 | 2024-12-09 14:15:00 | 109.64 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-12-11 12:30:00 | 109.67 | 2024-12-16 10:15:00 | 110.46 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-19 10:15:00 | 111.55 | 2024-12-20 10:15:00 | 110.90 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-01-07 09:15:00 | 122.08 | 2025-01-09 10:15:00 | 119.89 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-01-08 10:00:00 | 120.98 | 2025-01-09 10:15:00 | 119.89 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-01-08 11:00:00 | 121.02 | 2025-01-09 10:15:00 | 119.89 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-01-08 12:30:00 | 121.25 | 2025-01-09 10:15:00 | 119.89 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-01-21 09:15:00 | 118.22 | 2025-01-22 10:15:00 | 130.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-21 11:30:00 | 116.08 | 2025-01-22 10:15:00 | 127.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-29 12:45:00 | 113.68 | 2025-01-29 15:15:00 | 114.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-02-03 15:15:00 | 119.10 | 2025-02-04 09:15:00 | 113.53 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-02-13 11:30:00 | 100.15 | 2025-02-14 14:15:00 | 102.03 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-02-13 12:00:00 | 99.75 | 2025-02-14 14:15:00 | 102.03 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-02-14 09:15:00 | 97.82 | 2025-02-14 14:15:00 | 102.03 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2025-02-17 09:15:00 | 99.69 | 2025-02-17 11:15:00 | 99.55 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-03-04 09:15:00 | 89.33 | 2025-03-04 09:15:00 | 93.10 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2025-03-07 09:15:00 | 94.54 | 2025-03-10 10:15:00 | 92.76 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-03-12 11:30:00 | 91.34 | 2025-03-13 14:15:00 | 92.92 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-03-12 13:30:00 | 91.50 | 2025-03-13 14:15:00 | 92.92 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-03-27 15:15:00 | 102.21 | 2025-04-02 09:15:00 | 112.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-02 14:30:00 | 144.40 | 2025-05-05 09:15:00 | 146.28 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-05 12:15:00 | 143.85 | 2025-05-05 14:15:00 | 146.84 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-05-08 12:45:00 | 141.49 | 2025-05-09 09:15:00 | 134.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 12:45:00 | 141.49 | 2025-05-09 11:15:00 | 140.57 | STOP_HIT | 0.50 | 0.65% |
| BUY | retest2 | 2025-05-22 14:45:00 | 155.99 | 2025-05-29 09:15:00 | 171.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 12:15:00 | 155.45 | 2025-05-29 09:15:00 | 171.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-12 09:30:00 | 177.70 | 2025-06-12 14:15:00 | 173.26 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-06-17 11:15:00 | 171.24 | 2025-06-20 09:15:00 | 162.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 171.24 | 2025-06-20 10:15:00 | 165.80 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-06-30 13:00:00 | 160.00 | 2025-07-01 09:15:00 | 164.64 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-07-07 09:15:00 | 161.61 | 2025-07-07 11:15:00 | 158.92 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-07 10:45:00 | 160.70 | 2025-07-07 11:15:00 | 158.92 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-11 12:30:00 | 161.70 | 2025-07-15 10:15:00 | 177.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-01 12:30:00 | 217.99 | 2025-08-11 12:15:00 | 225.87 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-08-01 14:15:00 | 218.31 | 2025-08-11 12:15:00 | 225.87 | STOP_HIT | 1.00 | 3.46% |
| BUY | retest2 | 2025-08-04 09:15:00 | 220.61 | 2025-08-11 12:15:00 | 225.87 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest1 | 2025-08-18 12:00:00 | 199.20 | 2025-08-19 09:15:00 | 207.33 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2025-08-26 10:45:00 | 225.70 | 2025-08-29 09:15:00 | 218.06 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-08-28 09:30:00 | 227.27 | 2025-08-29 09:15:00 | 218.06 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-08-28 10:30:00 | 226.80 | 2025-08-29 09:15:00 | 218.06 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-09-11 15:00:00 | 181.07 | 2025-09-12 11:15:00 | 172.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 15:00:00 | 181.07 | 2025-09-16 11:15:00 | 169.34 | STOP_HIT | 0.50 | 6.48% |
| BUY | retest2 | 2025-09-23 12:45:00 | 175.76 | 2025-09-23 13:15:00 | 173.90 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-03 14:45:00 | 193.61 | 2025-10-07 12:15:00 | 189.65 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-03 15:15:00 | 193.65 | 2025-10-07 12:15:00 | 189.65 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-10-17 12:15:00 | 177.39 | 2025-10-17 12:15:00 | 175.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-10-27 12:45:00 | 171.80 | 2025-11-06 13:15:00 | 163.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:45:00 | 171.43 | 2025-11-06 13:15:00 | 162.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 170.70 | 2025-11-06 13:15:00 | 162.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 12:45:00 | 171.80 | 2025-11-07 09:15:00 | 168.61 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-10-29 09:45:00 | 171.43 | 2025-11-07 09:15:00 | 168.61 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2025-10-31 09:15:00 | 170.70 | 2025-11-07 09:15:00 | 168.61 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-12-09 14:15:00 | 153.22 | 2025-12-09 14:15:00 | 155.61 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-31 15:15:00 | 165.10 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-01-01 12:45:00 | 165.00 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-01 15:00:00 | 164.95 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-01-02 09:15:00 | 165.26 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-01-08 10:15:00 | 155.40 | 2026-01-09 14:15:00 | 147.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:15:00 | 155.40 | 2026-01-12 15:15:00 | 146.45 | STOP_HIT | 0.50 | 5.76% |
| SELL | retest2 | 2026-02-05 14:00:00 | 124.85 | 2026-02-09 14:15:00 | 129.49 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-02-05 15:15:00 | 124.85 | 2026-02-09 14:15:00 | 129.49 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest1 | 2026-02-12 09:30:00 | 120.37 | 2026-02-13 11:15:00 | 124.78 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2026-02-13 09:45:00 | 122.30 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-13 11:00:00 | 122.06 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-16 09:30:00 | 122.20 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-16 10:30:00 | 121.91 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-24 09:30:00 | 115.18 | 2026-02-26 11:15:00 | 117.01 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-24 11:45:00 | 115.58 | 2026-02-26 11:15:00 | 117.01 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-25 15:15:00 | 116.36 | 2026-02-26 11:15:00 | 117.01 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-03-19 12:30:00 | 113.25 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-19 14:30:00 | 113.60 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-03-19 15:15:00 | 114.20 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 12:15:00 | 113.18 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-04-02 11:15:00 | 111.10 | 2026-04-08 11:15:00 | 122.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 10:45:00 | 128.30 | 2026-05-05 10:15:00 | 121.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-04 11:45:00 | 128.29 | 2026-05-05 10:15:00 | 121.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-04 10:45:00 | 128.30 | 2026-05-06 09:15:00 | 123.86 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-05-04 11:45:00 | 128.29 | 2026-05-06 09:15:00 | 123.86 | STOP_HIT | 0.50 | 3.45% |
