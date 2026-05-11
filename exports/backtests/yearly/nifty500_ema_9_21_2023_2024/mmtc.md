# MMTC Ltd. (MMTC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 68.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 191 |
| ALERT1 | 138 |
| ALERT2 | 136 |
| ALERT2_SKIP | 89 |
| ALERT3 | 264 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 114 |
| PARTIAL | 25 |
| TARGET_HIT | 9 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 70 / 70
- **Target hits / Stop hits / Partials:** 9 / 106 / 25
- **Avg / median % per leg:** 0.54% / 0.02%
- **Sum % (uncompounded):** 75.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 15 | 41.7% | 5 | 30 | 1 | 0.53% | 19.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.63% | 7.3% |
| BUY @ 3rd Alert (retest2) | 34 | 13 | 38.2% | 5 | 29 | 0 | 0.34% | 11.7% |
| SELL (all) | 104 | 55 | 52.9% | 4 | 76 | 24 | 0.54% | 56.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -10.13% | -10.1% |
| SELL @ 3rd Alert (retest2) | 103 | 55 | 53.4% | 4 | 75 | 24 | 0.65% | 66.8% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | -0.96% | -2.9% |
| retest2 (combined) | 137 | 68 | 49.6% | 9 | 104 | 24 | 0.57% | 78.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 30.10 | 29.83 | 29.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 30.85 | 30.03 | 29.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 15:15:00 | 30.35 | 30.39 | 30.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 30.00 | 30.32 | 30.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 30.00 | 30.32 | 30.15 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 09:15:00 | 30.05 | 30.09 | 30.09 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 10:15:00 | 30.10 | 30.09 | 30.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 13:15:00 | 30.20 | 30.12 | 30.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 09:15:00 | 30.05 | 30.14 | 30.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 30.05 | 30.14 | 30.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 30.05 | 30.14 | 30.12 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 11:15:00 | 29.95 | 30.08 | 30.10 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 09:15:00 | 30.35 | 30.10 | 30.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 11:15:00 | 30.70 | 30.44 | 30.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 10:15:00 | 30.55 | 30.60 | 30.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 13:15:00 | 30.50 | 30.59 | 30.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 13:15:00 | 30.50 | 30.59 | 30.49 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 13:15:00 | 31.75 | 31.81 | 31.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 13:15:00 | 31.65 | 31.74 | 31.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 31.40 | 31.38 | 31.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 31.40 | 31.38 | 31.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 31.40 | 31.38 | 31.52 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 13:15:00 | 32.70 | 31.72 | 31.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 14:15:00 | 33.75 | 32.53 | 32.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 14:15:00 | 33.35 | 33.40 | 32.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 32.85 | 33.18 | 32.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 32.85 | 33.18 | 32.97 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 32.00 | 32.79 | 32.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 14:15:00 | 31.75 | 31.92 | 32.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 09:15:00 | 32.00 | 31.91 | 32.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 32.00 | 31.91 | 32.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 32.00 | 31.91 | 32.05 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 10:15:00 | 32.20 | 31.81 | 31.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 09:15:00 | 33.85 | 32.33 | 32.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 12:15:00 | 33.15 | 33.30 | 32.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 33.10 | 33.23 | 32.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 33.10 | 33.23 | 32.96 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 32.30 | 32.76 | 32.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 11:15:00 | 32.00 | 32.48 | 32.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 32.65 | 32.37 | 32.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 32.65 | 32.37 | 32.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 32.65 | 32.37 | 32.51 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 11:15:00 | 33.50 | 32.47 | 32.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 14:15:00 | 34.35 | 33.20 | 32.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 34.35 | 34.48 | 33.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 33.85 | 34.34 | 33.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 33.85 | 34.34 | 33.98 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 12:15:00 | 33.45 | 33.88 | 33.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 14:15:00 | 33.25 | 33.44 | 33.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 12:15:00 | 33.90 | 33.46 | 33.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 12:15:00 | 33.90 | 33.46 | 33.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 12:15:00 | 33.90 | 33.46 | 33.54 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 14:15:00 | 34.30 | 33.71 | 33.65 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 11:15:00 | 33.55 | 33.76 | 33.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 12:15:00 | 33.45 | 33.69 | 33.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 33.55 | 33.55 | 33.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 10:15:00 | 33.90 | 33.62 | 33.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 33.90 | 33.62 | 33.67 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 12:15:00 | 33.90 | 33.70 | 33.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 13:15:00 | 34.00 | 33.76 | 33.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 14:15:00 | 33.60 | 33.73 | 33.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 14:15:00 | 33.60 | 33.73 | 33.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 33.60 | 33.73 | 33.72 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 36.40 | 37.07 | 37.14 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 10:15:00 | 37.90 | 37.19 | 37.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 38.45 | 37.61 | 37.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 15:15:00 | 37.70 | 37.87 | 37.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 15:15:00 | 37.70 | 37.87 | 37.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 37.70 | 37.87 | 37.65 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 09:15:00 | 36.50 | 37.35 | 37.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 10:15:00 | 36.25 | 37.13 | 37.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 36.55 | 36.50 | 36.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 36.55 | 36.50 | 36.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 36.55 | 36.50 | 36.87 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 12:15:00 | 38.00 | 37.15 | 37.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 09:15:00 | 38.35 | 37.76 | 37.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 37.90 | 38.01 | 37.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 13:15:00 | 37.90 | 38.01 | 37.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 37.90 | 38.01 | 37.69 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 38.35 | 38.88 | 38.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 37.90 | 38.68 | 38.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 38.60 | 38.39 | 38.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 10:15:00 | 38.60 | 38.39 | 38.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 38.60 | 38.39 | 38.60 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 39.90 | 38.76 | 38.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 42.35 | 39.94 | 39.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 12:15:00 | 42.50 | 42.62 | 41.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 41.75 | 42.39 | 41.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 41.75 | 42.39 | 41.82 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 42.25 | 42.44 | 42.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 42.00 | 42.35 | 42.42 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 43.10 | 42.41 | 42.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 44.05 | 42.87 | 42.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 09:15:00 | 66.20 | 67.65 | 64.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 15:15:00 | 65.00 | 66.48 | 65.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 65.00 | 66.48 | 65.35 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 58.75 | 64.42 | 64.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 56.60 | 59.54 | 61.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 59.25 | 59.23 | 61.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 61.90 | 59.82 | 60.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 61.90 | 59.82 | 60.76 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 61.85 | 61.28 | 61.21 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 60.35 | 61.29 | 61.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 10:15:00 | 59.85 | 61.00 | 61.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 59.45 | 59.34 | 59.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 11:15:00 | 59.10 | 59.29 | 59.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 59.10 | 59.29 | 59.75 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 12:15:00 | 59.85 | 58.91 | 58.79 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 14:15:00 | 58.40 | 58.85 | 58.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 09:15:00 | 57.60 | 58.54 | 58.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 09:15:00 | 58.40 | 58.12 | 58.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 58.40 | 58.12 | 58.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 58.40 | 58.12 | 58.37 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 15:15:00 | 58.50 | 58.27 | 58.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 09:15:00 | 60.35 | 58.68 | 58.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 58.25 | 59.23 | 58.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 58.25 | 59.23 | 58.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 58.25 | 59.23 | 58.95 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 58.10 | 58.78 | 58.79 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 15:15:00 | 58.95 | 58.80 | 58.80 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 09:15:00 | 58.25 | 58.69 | 58.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 15:15:00 | 58.10 | 58.54 | 58.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 58.85 | 58.61 | 58.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 58.85 | 58.61 | 58.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 58.85 | 58.61 | 58.65 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 59.30 | 58.74 | 58.68 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 56.65 | 58.32 | 58.50 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 58.90 | 57.98 | 57.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 63.80 | 59.72 | 58.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 09:15:00 | 78.40 | 84.10 | 80.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 78.40 | 84.10 | 80.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 78.40 | 84.10 | 80.99 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 70.60 | 78.04 | 79.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 64.05 | 70.85 | 74.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-30 12:15:00 | 54.45 | 53.24 | 54.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 09:15:00 | 57.15 | 54.49 | 54.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 57.15 | 54.49 | 54.93 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 55.55 | 55.27 | 55.24 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 54.40 | 55.17 | 55.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 53.70 | 54.62 | 54.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 11:15:00 | 54.50 | 54.14 | 54.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 11:15:00 | 54.50 | 54.14 | 54.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 54.50 | 54.14 | 54.48 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 54.85 | 54.24 | 54.21 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 11:15:00 | 53.40 | 54.11 | 54.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 52.80 | 53.06 | 53.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 53.55 | 52.25 | 52.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 53.55 | 52.25 | 52.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 53.55 | 52.25 | 52.49 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 13:15:00 | 53.00 | 52.68 | 52.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 15:15:00 | 53.20 | 52.80 | 52.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 12:15:00 | 52.10 | 52.72 | 52.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 12:15:00 | 52.10 | 52.72 | 52.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 12:15:00 | 52.10 | 52.72 | 52.70 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 13:15:00 | 52.10 | 52.60 | 52.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 12:15:00 | 51.95 | 52.22 | 52.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 51.80 | 51.65 | 51.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 51.80 | 51.65 | 51.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 51.80 | 51.65 | 51.88 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 52.10 | 51.96 | 51.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 52.50 | 52.07 | 52.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 12:15:00 | 52.55 | 52.57 | 52.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 12:15:00 | 52.55 | 52.57 | 52.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 52.55 | 52.57 | 52.29 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 15:15:00 | 51.50 | 52.34 | 52.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 50.95 | 51.68 | 51.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 12:15:00 | 50.30 | 50.29 | 50.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 50.75 | 50.37 | 50.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 50.75 | 50.37 | 50.58 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 12:15:00 | 51.10 | 50.75 | 50.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 13:15:00 | 52.45 | 51.09 | 50.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 51.50 | 51.61 | 51.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 51.50 | 51.61 | 51.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 51.50 | 51.61 | 51.37 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 09:15:00 | 50.85 | 51.29 | 51.31 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 55.15 | 51.74 | 51.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 10:15:00 | 55.75 | 52.54 | 51.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 10:15:00 | 57.85 | 57.93 | 56.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 13:15:00 | 57.20 | 57.64 | 56.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 57.20 | 57.64 | 56.60 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 59.65 | 60.36 | 60.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 56.45 | 59.58 | 60.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 57.85 | 57.67 | 58.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 58.80 | 57.94 | 58.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 58.80 | 57.94 | 58.52 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 59.75 | 58.84 | 58.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 12:15:00 | 61.75 | 59.46 | 59.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 59.60 | 59.97 | 59.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 59.60 | 59.97 | 59.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 59.60 | 59.97 | 59.56 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 59.65 | 59.95 | 59.95 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 60.60 | 59.91 | 59.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 10:15:00 | 61.85 | 60.29 | 60.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 63.30 | 63.58 | 62.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 63.55 | 63.56 | 62.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 63.55 | 63.56 | 62.79 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 67.00 | 68.22 | 68.25 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 68.60 | 68.21 | 68.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 72.70 | 69.10 | 68.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 72.50 | 73.15 | 71.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 72.50 | 73.15 | 71.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 72.50 | 73.15 | 71.40 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 69.80 | 70.91 | 70.99 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 73.95 | 71.20 | 71.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 74.90 | 71.94 | 71.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 15:15:00 | 74.95 | 75.45 | 74.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 15:15:00 | 79.20 | 79.80 | 78.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 79.20 | 79.80 | 78.72 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 90.30 | 92.60 | 92.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 85.70 | 90.53 | 91.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 15:15:00 | 77.30 | 77.02 | 79.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 09:15:00 | 80.55 | 77.73 | 79.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 80.55 | 77.73 | 79.22 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 15:15:00 | 80.55 | 79.81 | 79.80 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 09:15:00 | 77.25 | 79.30 | 79.57 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 09:15:00 | 82.20 | 79.54 | 79.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 85.55 | 82.31 | 81.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 85.50 | 86.35 | 84.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 14:15:00 | 85.10 | 86.10 | 84.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 85.10 | 86.10 | 84.78 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 13:15:00 | 83.80 | 84.24 | 84.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 14:15:00 | 82.55 | 83.31 | 83.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 81.25 | 79.61 | 80.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 81.25 | 79.61 | 80.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 81.25 | 79.61 | 80.63 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 11:15:00 | 79.05 | 78.21 | 78.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 13:15:00 | 80.15 | 78.79 | 78.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 76.85 | 78.98 | 78.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 76.85 | 78.98 | 78.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 76.85 | 78.98 | 78.70 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 76.65 | 78.14 | 78.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 10:15:00 | 76.20 | 76.95 | 77.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 68.40 | 67.22 | 69.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 68.80 | 67.54 | 69.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 68.80 | 67.54 | 69.00 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 66.05 | 64.97 | 64.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 66.75 | 65.75 | 65.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 66.25 | 66.35 | 65.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 65.80 | 66.21 | 65.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 65.80 | 66.21 | 65.86 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 13:15:00 | 65.40 | 65.67 | 65.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 14:15:00 | 64.95 | 65.53 | 65.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 65.35 | 65.30 | 65.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 65.35 | 65.30 | 65.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 65.35 | 65.30 | 65.48 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 13:15:00 | 67.55 | 65.81 | 65.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 09:15:00 | 68.65 | 66.57 | 66.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 66.60 | 67.03 | 66.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 14:15:00 | 66.60 | 67.03 | 66.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 66.60 | 67.03 | 66.54 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 15:15:00 | 75.25 | 75.80 | 75.87 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 76.85 | 76.01 | 75.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 13:15:00 | 77.15 | 76.49 | 76.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 15:15:00 | 76.35 | 76.51 | 76.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 15:15:00 | 76.35 | 76.51 | 76.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 76.35 | 76.51 | 76.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 76.15 | 76.51 | 76.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 75.50 | 76.31 | 76.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 75.75 | 76.31 | 76.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 76.05 | 76.26 | 76.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 11:30:00 | 76.25 | 76.19 | 76.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 12:15:00 | 75.35 | 76.02 | 76.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 75.35 | 76.02 | 76.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 72.75 | 75.17 | 75.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 73.35 | 73.08 | 74.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:00:00 | 73.35 | 73.08 | 74.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 73.65 | 73.19 | 74.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 72.80 | 73.19 | 74.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 69.16 | 71.71 | 72.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 71.55 | 71.38 | 72.17 | SL hit (close>ema200) qty=0.50 sl=71.38 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 72.60 | 72.25 | 72.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 73.85 | 72.65 | 72.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 10:15:00 | 75.00 | 75.10 | 74.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:45:00 | 74.75 | 75.10 | 74.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 74.70 | 74.99 | 74.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:00:00 | 74.70 | 74.99 | 74.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 74.90 | 74.89 | 74.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 74.90 | 74.89 | 74.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 77.35 | 75.38 | 74.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:45:00 | 78.00 | 75.81 | 75.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 13:30:00 | 77.75 | 76.47 | 75.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 14:15:00 | 77.80 | 76.47 | 75.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 14:45:00 | 77.55 | 76.53 | 75.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 75.80 | 76.09 | 75.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:00:00 | 75.80 | 76.09 | 75.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 75.75 | 76.02 | 75.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 15:15:00 | 75.75 | 76.02 | 75.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 75.75 | 75.97 | 75.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 75.35 | 75.82 | 75.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 74.85 | 75.63 | 75.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 74.85 | 75.63 | 75.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 74.20 | 75.08 | 75.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 10:15:00 | 75.25 | 74.95 | 75.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 10:15:00 | 75.25 | 74.95 | 75.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 75.25 | 74.95 | 75.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:00:00 | 75.25 | 74.95 | 75.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 75.50 | 75.06 | 75.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 14:15:00 | 74.80 | 75.17 | 75.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 15:00:00 | 74.80 | 75.10 | 75.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:30:00 | 74.40 | 74.96 | 75.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 71.06 | 72.15 | 73.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 71.06 | 72.15 | 73.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 70.68 | 72.15 | 73.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 71.30 | 71.25 | 72.19 | SL hit (close>ema200) qty=0.50 sl=71.25 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 70.85 | 69.66 | 69.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 71.05 | 70.12 | 69.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 72.40 | 72.47 | 71.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 72.40 | 72.47 | 71.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 71.70 | 72.32 | 71.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 71.70 | 72.32 | 71.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 71.85 | 72.23 | 71.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:30:00 | 71.50 | 72.23 | 71.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 71.60 | 72.03 | 71.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 71.60 | 72.03 | 71.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 71.80 | 71.99 | 71.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 72.10 | 71.99 | 71.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 74.10 | 75.17 | 75.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 74.10 | 75.17 | 75.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 12:15:00 | 73.60 | 74.02 | 74.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 14:15:00 | 72.05 | 71.72 | 72.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 15:00:00 | 72.05 | 71.72 | 72.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 71.65 | 71.71 | 72.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 71.50 | 71.71 | 72.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 70.80 | 71.53 | 72.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:30:00 | 70.35 | 71.09 | 71.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:30:00 | 70.10 | 70.75 | 71.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:15:00 | 70.20 | 70.67 | 71.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:30:00 | 70.45 | 70.62 | 71.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 73.60 | 71.17 | 71.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 73.60 | 71.17 | 71.26 | SL hit (close>static) qty=1.00 sl=72.40 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 72.50 | 71.43 | 71.37 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 68.25 | 71.54 | 71.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 66.75 | 70.58 | 71.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 68.60 | 68.07 | 69.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 68.60 | 68.07 | 69.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 68.55 | 68.17 | 69.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 68.80 | 68.17 | 69.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 69.45 | 68.27 | 68.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 69.45 | 68.27 | 68.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 69.90 | 68.60 | 69.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 69.95 | 68.60 | 69.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 68.30 | 68.54 | 69.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:30:00 | 68.10 | 68.41 | 68.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:00:00 | 68.20 | 68.37 | 68.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 70.15 | 68.93 | 68.95 | SL hit (close>static) qty=1.00 sl=70.10 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 11:15:00 | 69.70 | 69.09 | 69.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 70.25 | 69.61 | 69.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 76.75 | 76.76 | 75.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 76.75 | 76.76 | 75.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 76.49 | 76.92 | 76.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:30:00 | 76.35 | 76.92 | 76.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 76.10 | 76.76 | 76.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 76.81 | 76.76 | 76.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 75.83 | 76.57 | 76.25 | SL hit (close<static) qty=1.00 sl=76.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 80.95 | 81.80 | 81.84 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 84.23 | 82.15 | 81.98 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 81.55 | 82.70 | 82.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 15:15:00 | 81.43 | 81.97 | 82.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 79.65 | 79.59 | 80.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 14:00:00 | 79.65 | 79.59 | 80.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 79.39 | 79.30 | 79.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:45:00 | 79.52 | 79.30 | 79.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 79.23 | 78.52 | 79.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:45:00 | 79.06 | 78.52 | 79.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 79.10 | 78.64 | 79.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 79.74 | 78.64 | 79.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 83.40 | 79.59 | 79.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 86.53 | 80.98 | 80.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 84.31 | 84.76 | 83.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:45:00 | 84.19 | 84.76 | 83.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 85.70 | 85.40 | 84.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:15:00 | 86.34 | 85.40 | 84.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 83.89 | 84.98 | 84.71 | SL hit (close<static) qty=1.00 sl=84.03 alert=retest2 |

### Cycle 80 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 83.51 | 84.45 | 84.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 82.36 | 83.84 | 84.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 83.00 | 82.93 | 83.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 15:00:00 | 83.00 | 82.93 | 83.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 84.04 | 83.03 | 83.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 84.04 | 83.03 | 83.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 84.24 | 83.27 | 83.50 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 85.41 | 83.89 | 83.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 13:15:00 | 86.51 | 84.41 | 84.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 88.40 | 89.14 | 87.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 15:00:00 | 88.40 | 89.14 | 87.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 87.84 | 88.79 | 87.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 87.03 | 88.79 | 87.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 95.77 | 90.19 | 88.09 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 87.76 | 90.59 | 90.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 87.21 | 89.91 | 90.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 90.10 | 88.26 | 89.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 90.10 | 88.26 | 89.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 90.10 | 88.26 | 89.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 90.10 | 88.26 | 89.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 88.41 | 88.29 | 89.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 88.15 | 88.37 | 89.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:45:00 | 88.38 | 88.31 | 88.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 83.74 | 86.80 | 87.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 83.96 | 86.80 | 87.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 95.07 | 87.95 | 88.01 | SL hit (close>ema200) qty=0.50 sl=87.95 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 99.00 | 90.16 | 89.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 101.25 | 92.38 | 90.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 108.30 | 113.52 | 106.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 108.30 | 113.52 | 106.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 108.30 | 113.52 | 106.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 108.30 | 113.52 | 106.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 106.88 | 109.73 | 106.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:45:00 | 105.38 | 109.73 | 106.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 106.85 | 109.16 | 106.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 107.91 | 109.16 | 106.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 107.00 | 108.72 | 106.93 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 15:15:00 | 105.10 | 106.01 | 106.13 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 114.71 | 107.75 | 106.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 10:15:00 | 114.82 | 109.17 | 107.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 110.09 | 110.79 | 109.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 110.09 | 110.79 | 109.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 110.09 | 110.79 | 109.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 110.11 | 110.79 | 109.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 108.92 | 110.09 | 109.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 108.92 | 110.09 | 109.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 110.35 | 110.14 | 109.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 15:15:00 | 111.60 | 110.14 | 109.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 107.10 | 109.29 | 109.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 107.10 | 109.29 | 109.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 106.13 | 108.23 | 108.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 103.03 | 101.71 | 104.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 103.03 | 101.71 | 104.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 103.03 | 101.71 | 104.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 103.03 | 101.71 | 104.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 104.46 | 102.26 | 104.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 102.20 | 102.37 | 103.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 102.25 | 102.11 | 102.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 97.09 | 99.56 | 100.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 97.14 | 99.56 | 100.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 99.17 | 99.10 | 100.10 | SL hit (close>ema200) qty=0.50 sl=99.10 alert=retest2 |

### Cycle 87 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 105.90 | 101.50 | 101.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 10:15:00 | 106.90 | 102.58 | 101.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 103.00 | 103.62 | 102.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 103.00 | 103.62 | 102.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 103.90 | 103.68 | 102.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 101.45 | 103.68 | 102.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 101.09 | 103.16 | 102.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 99.75 | 103.16 | 102.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 99.67 | 102.46 | 102.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 99.67 | 102.46 | 102.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 100.34 | 102.04 | 102.05 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 105.10 | 101.18 | 101.15 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 103.15 | 103.88 | 103.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 100.89 | 103.28 | 103.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 101.39 | 101.38 | 102.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 10:15:00 | 101.39 | 101.38 | 102.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 101.39 | 101.38 | 102.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 102.04 | 101.38 | 102.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 101.01 | 101.36 | 102.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 100.86 | 101.28 | 101.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 104.21 | 101.90 | 102.08 | SL hit (close>static) qty=1.00 sl=102.17 alert=retest2 |

### Cycle 91 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 104.14 | 102.35 | 102.27 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 101.00 | 102.36 | 102.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 15:15:00 | 100.45 | 101.52 | 101.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 98.90 | 98.57 | 99.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 98.90 | 98.57 | 99.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 98.90 | 98.57 | 99.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:30:00 | 98.23 | 98.53 | 99.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 13:00:00 | 98.38 | 98.48 | 99.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:30:00 | 98.37 | 98.47 | 98.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 15:00:00 | 98.25 | 98.47 | 98.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 95.95 | 97.89 | 98.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 93.01 | 96.30 | 97.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 93.32 | 95.73 | 97.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 93.46 | 95.73 | 97.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 93.45 | 95.73 | 97.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 93.34 | 95.73 | 97.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 95.43 | 94.62 | 95.64 | SL hit (close>ema200) qty=0.50 sl=94.62 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 95.83 | 94.83 | 94.80 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 94.36 | 94.90 | 94.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 93.76 | 94.28 | 94.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 91.11 | 90.55 | 91.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:45:00 | 90.91 | 90.55 | 91.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 91.43 | 90.73 | 91.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 91.50 | 90.73 | 91.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 91.39 | 90.98 | 91.56 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 92.50 | 91.85 | 91.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 94.37 | 92.35 | 92.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 92.70 | 92.79 | 92.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:15:00 | 92.73 | 92.79 | 92.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 92.26 | 92.69 | 92.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:30:00 | 92.13 | 92.69 | 92.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 92.37 | 92.62 | 92.36 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 90.92 | 92.08 | 92.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 90.83 | 91.68 | 91.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 91.25 | 90.59 | 91.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 91.25 | 90.59 | 91.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 91.25 | 90.59 | 91.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 90.61 | 90.59 | 91.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 91.88 | 90.85 | 91.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:30:00 | 91.90 | 90.85 | 91.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 89.65 | 90.65 | 90.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:15:00 | 89.18 | 89.78 | 90.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 88.83 | 90.08 | 90.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 84.72 | 87.60 | 88.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 84.39 | 87.60 | 88.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-08 09:15:00 | 80.26 | 82.81 | 84.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 85.91 | 85.10 | 85.00 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 85.11 | 85.16 | 85.16 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 85.67 | 85.26 | 85.21 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 84.96 | 85.15 | 85.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 84.63 | 85.01 | 85.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 15:15:00 | 83.80 | 83.75 | 84.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:15:00 | 84.39 | 83.75 | 84.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 84.66 | 83.93 | 84.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:45:00 | 82.89 | 83.87 | 84.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 13:45:00 | 82.79 | 82.60 | 82.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 78.75 | 79.90 | 81.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 78.65 | 79.90 | 81.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 74.60 | 76.99 | 78.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 75.65 | 75.12 | 75.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 78.14 | 75.72 | 75.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 77.34 | 78.76 | 78.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 77.34 | 78.76 | 78.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 77.34 | 78.76 | 78.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 77.34 | 78.76 | 78.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 77.23 | 78.46 | 78.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 77.23 | 78.46 | 78.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 77.54 | 78.14 | 77.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:45:00 | 77.39 | 78.14 | 77.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 77.05 | 77.67 | 77.75 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 79.18 | 78.00 | 77.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 80.53 | 78.85 | 78.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 15:15:00 | 80.50 | 80.74 | 79.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:15:00 | 79.52 | 80.74 | 79.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 81.38 | 80.87 | 80.09 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 78.39 | 79.87 | 79.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 77.95 | 78.74 | 79.25 | Break + close below crossover candle low |

### Cycle 105 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 84.49 | 79.81 | 79.61 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 77.21 | 79.38 | 79.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 75.00 | 77.42 | 78.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 15:15:00 | 75.80 | 75.54 | 76.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 15:15:00 | 75.80 | 75.54 | 76.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 75.80 | 75.54 | 76.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 77.21 | 75.54 | 76.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 77.39 | 75.91 | 76.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 77.39 | 75.91 | 76.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 77.20 | 76.17 | 76.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:30:00 | 76.71 | 76.40 | 76.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:00:00 | 76.69 | 76.40 | 76.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:30:00 | 76.67 | 76.37 | 76.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:45:00 | 76.28 | 74.59 | 74.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 76.46 | 74.97 | 74.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 76.46 | 74.97 | 74.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 78.32 | 76.31 | 75.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 77.90 | 78.31 | 77.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 77.90 | 78.31 | 77.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 77.90 | 78.31 | 77.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 80.04 | 78.30 | 77.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 79.10 | 79.04 | 78.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:00:00 | 78.69 | 78.56 | 78.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 11:00:00 | 79.76 | 78.70 | 78.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 80.21 | 80.86 | 80.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:00:00 | 80.21 | 80.86 | 80.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 80.69 | 80.73 | 80.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:45:00 | 80.00 | 80.73 | 80.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 81.16 | 80.76 | 80.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 81.87 | 80.46 | 80.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:30:00 | 82.09 | 81.45 | 80.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 81.89 | 82.43 | 82.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 81.89 | 82.43 | 82.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 81.00 | 82.14 | 82.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 80.08 | 79.87 | 80.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 80.08 | 79.87 | 80.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 80.08 | 79.87 | 80.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 79.19 | 79.71 | 80.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 75.23 | 76.84 | 77.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 73.18 | 73.12 | 73.78 | SL hit (close>ema200) qty=0.50 sl=73.12 alert=retest2 |

### Cycle 109 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 73.91 | 72.58 | 72.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 74.17 | 73.29 | 72.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 11:15:00 | 73.45 | 73.78 | 73.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 11:15:00 | 73.45 | 73.78 | 73.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 73.45 | 73.78 | 73.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 73.56 | 73.78 | 73.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 73.61 | 73.74 | 73.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 73.98 | 73.78 | 73.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 73.06 | 74.19 | 73.98 | SL hit (close<static) qty=1.00 sl=73.33 alert=retest2 |

### Cycle 110 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 71.66 | 73.68 | 73.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 71.50 | 72.92 | 73.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 71.04 | 71.02 | 71.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 15:15:00 | 70.50 | 71.02 | 71.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 10:15:00 | 77.64 | 72.21 | 72.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 111 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 77.64 | 72.21 | 72.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 11:15:00 | 79.00 | 73.57 | 72.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 74.90 | 75.09 | 73.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:15:00 | 74.31 | 75.09 | 73.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 74.25 | 74.92 | 73.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:45:00 | 74.00 | 74.92 | 73.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 74.67 | 74.87 | 74.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:30:00 | 73.83 | 74.87 | 74.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 74.30 | 74.76 | 74.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:30:00 | 74.16 | 74.76 | 74.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 74.11 | 74.63 | 74.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:30:00 | 74.20 | 74.63 | 74.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 74.70 | 74.64 | 74.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:30:00 | 74.41 | 74.64 | 74.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 74.30 | 74.57 | 74.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 72.06 | 74.57 | 74.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 71.60 | 73.98 | 73.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 71.60 | 73.98 | 73.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 72.16 | 73.62 | 73.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 70.56 | 72.40 | 73.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 68.29 | 68.28 | 69.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 68.29 | 68.28 | 69.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 68.79 | 68.52 | 69.28 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 70.05 | 69.46 | 69.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 72.50 | 70.41 | 69.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 12:15:00 | 71.87 | 71.96 | 71.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 13:15:00 | 71.91 | 71.96 | 71.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 71.84 | 71.94 | 71.55 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 69.56 | 71.31 | 71.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 68.93 | 69.82 | 70.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 14:15:00 | 69.60 | 69.46 | 69.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 15:00:00 | 69.60 | 69.46 | 69.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 68.66 | 69.30 | 69.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 67.10 | 69.30 | 69.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 63.74 | 66.01 | 67.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 66.02 | 65.87 | 67.16 | SL hit (close>ema200) qty=0.50 sl=65.87 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 68.45 | 67.02 | 66.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 13:15:00 | 69.15 | 68.38 | 67.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 68.78 | 69.11 | 68.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 68.78 | 69.11 | 68.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 68.78 | 69.11 | 68.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 68.41 | 69.11 | 68.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 67.85 | 68.86 | 68.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 68.16 | 68.86 | 68.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 68.41 | 68.77 | 68.42 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 67.14 | 68.13 | 68.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 66.25 | 66.86 | 67.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 67.05 | 66.90 | 67.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 67.05 | 66.90 | 67.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 67.05 | 66.90 | 67.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 66.79 | 66.90 | 67.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 68.46 | 67.25 | 67.30 | SL hit (close>static) qty=1.00 sl=68.09 alert=retest2 |

### Cycle 117 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 68.38 | 67.48 | 67.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 69.40 | 67.86 | 67.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 68.23 | 68.26 | 67.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 68.23 | 68.26 | 67.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 68.23 | 68.26 | 67.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 68.31 | 68.26 | 67.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 67.93 | 68.20 | 67.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 67.93 | 68.20 | 67.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 67.83 | 68.12 | 67.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 67.83 | 68.12 | 67.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 67.72 | 68.04 | 67.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 67.72 | 68.04 | 67.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 67.70 | 67.90 | 67.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 67.52 | 67.90 | 67.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 67.38 | 67.81 | 67.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 66.88 | 67.47 | 67.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 62.72 | 62.17 | 63.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 62.61 | 62.17 | 63.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 56.30 | 54.96 | 56.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 56.81 | 54.96 | 56.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 55.72 | 55.11 | 56.19 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 57.09 | 56.48 | 56.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 57.74 | 56.73 | 56.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 56.95 | 57.06 | 56.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 56.95 | 57.06 | 56.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 56.70 | 56.99 | 56.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 56.70 | 56.99 | 56.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 57.01 | 56.99 | 56.77 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 55.37 | 56.49 | 56.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 54.21 | 55.28 | 55.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 48.67 | 48.60 | 50.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:15:00 | 48.78 | 48.60 | 50.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 50.27 | 48.97 | 49.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 50.27 | 48.97 | 49.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 50.11 | 49.20 | 49.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:15:00 | 49.87 | 49.35 | 49.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 51.34 | 50.05 | 50.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 51.34 | 50.05 | 50.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 15:15:00 | 52.10 | 51.29 | 50.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 54.20 | 54.80 | 53.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 54.20 | 54.80 | 53.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 53.88 | 54.62 | 53.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 53.88 | 54.62 | 53.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 53.80 | 54.46 | 53.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 53.80 | 54.46 | 53.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 53.67 | 54.30 | 53.84 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 51.36 | 53.17 | 53.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 50.71 | 51.53 | 52.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 51.53 | 50.28 | 50.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 51.53 | 50.28 | 50.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 51.53 | 50.28 | 50.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 51.53 | 50.28 | 50.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 51.23 | 50.47 | 50.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 50.71 | 50.68 | 50.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 52.50 | 51.12 | 50.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 52.50 | 51.12 | 50.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 15:15:00 | 53.70 | 52.67 | 51.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 54.95 | 55.65 | 55.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 54.95 | 55.65 | 55.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 54.95 | 55.65 | 55.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 54.95 | 55.65 | 55.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 54.40 | 55.40 | 54.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 54.40 | 55.40 | 54.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 54.06 | 55.13 | 54.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 54.04 | 55.13 | 54.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 53.80 | 54.54 | 54.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 53.45 | 54.16 | 54.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 10:15:00 | 54.17 | 54.16 | 54.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 10:15:00 | 54.17 | 54.16 | 54.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 54.17 | 54.16 | 54.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 54.69 | 54.16 | 54.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 53.10 | 53.95 | 54.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 53.06 | 53.95 | 54.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:00:00 | 52.90 | 53.74 | 54.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 55.85 | 53.50 | 53.63 | SL hit (close>static) qty=1.00 sl=54.50 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 54.37 | 53.76 | 53.73 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 53.01 | 53.63 | 53.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 52.79 | 53.46 | 53.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 53.88 | 53.16 | 53.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 53.88 | 53.16 | 53.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 53.88 | 53.16 | 53.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 53.88 | 53.16 | 53.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 53.86 | 53.30 | 53.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 53.41 | 53.38 | 53.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 53.89 | 53.56 | 53.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 53.89 | 53.56 | 53.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 15:15:00 | 54.23 | 53.74 | 53.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 53.28 | 53.65 | 53.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 53.28 | 53.65 | 53.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 53.28 | 53.65 | 53.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 55.18 | 53.97 | 53.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:30:00 | 54.61 | 54.17 | 53.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 54.61 | 54.17 | 53.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:30:00 | 54.56 | 54.18 | 53.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 52.92 | 54.07 | 54.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 52.92 | 54.07 | 54.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-04 10:15:00 | 52.75 | 53.81 | 53.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 52.75 | 53.81 | 53.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 51.93 | 53.22 | 53.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 49.05 | 49.04 | 50.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 50.34 | 49.04 | 50.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 49.68 | 49.16 | 50.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 49.48 | 49.28 | 50.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 48.88 | 49.68 | 50.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 50.45 | 49.86 | 49.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 50.45 | 49.86 | 49.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 52.30 | 50.51 | 50.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 53.77 | 53.81 | 53.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 53.77 | 53.81 | 53.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 55.24 | 55.42 | 54.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 55.07 | 55.42 | 54.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 59.20 | 56.35 | 55.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:15:00 | 62.12 | 57.02 | 55.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 14:15:00 | 58.09 | 58.31 | 58.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 14:15:00 | 58.09 | 58.31 | 58.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 57.24 | 57.92 | 58.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 56.25 | 55.80 | 56.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 56.25 | 55.80 | 56.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 56.25 | 55.80 | 56.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 56.25 | 55.80 | 56.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 56.19 | 55.88 | 56.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 55.75 | 56.09 | 56.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 52.96 | 54.31 | 55.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 54.10 | 53.87 | 54.52 | SL hit (close>ema200) qty=0.50 sl=53.87 alert=retest2 |

### Cycle 131 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 56.04 | 53.65 | 53.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 56.50 | 55.12 | 54.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 61.50 | 61.63 | 60.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 61.50 | 61.63 | 60.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 61.12 | 61.53 | 60.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 60.98 | 61.53 | 60.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 62.10 | 61.64 | 60.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 65.75 | 61.56 | 61.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-28 14:15:00 | 72.33 | 68.04 | 65.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 77.18 | 78.27 | 78.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 14:15:00 | 77.05 | 78.02 | 78.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 79.10 | 76.75 | 77.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 79.10 | 76.75 | 77.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 79.10 | 76.75 | 77.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 79.10 | 76.75 | 77.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 78.09 | 77.02 | 77.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 76.84 | 77.24 | 77.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 12:15:00 | 73.00 | 74.87 | 75.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 74.50 | 74.39 | 75.35 | SL hit (close>ema200) qty=0.50 sl=74.39 alert=retest2 |

### Cycle 133 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 72.09 | 70.47 | 70.46 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 69.25 | 70.56 | 70.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 68.70 | 69.80 | 70.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 67.50 | 67.34 | 68.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 67.50 | 67.34 | 68.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 70.40 | 67.96 | 68.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 70.40 | 67.96 | 68.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 70.25 | 68.42 | 68.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 71.08 | 69.60 | 68.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 69.91 | 69.97 | 69.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 69.91 | 69.97 | 69.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 69.91 | 69.97 | 69.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:30:00 | 69.61 | 69.97 | 69.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 69.55 | 69.88 | 69.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 69.55 | 69.88 | 69.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 69.09 | 69.72 | 69.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 70.11 | 69.72 | 69.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 71.20 | 71.49 | 71.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 71.20 | 71.49 | 71.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 71.11 | 71.38 | 71.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 70.70 | 70.67 | 70.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 70.70 | 70.67 | 70.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 70.70 | 70.67 | 70.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 70.31 | 70.48 | 70.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 70.18 | 70.25 | 70.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 70.24 | 69.76 | 70.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 15:15:00 | 68.60 | 68.47 | 68.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 68.60 | 68.47 | 68.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 68.89 | 68.55 | 68.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 68.51 | 68.70 | 68.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 68.51 | 68.70 | 68.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 68.51 | 68.70 | 68.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 69.00 | 68.70 | 68.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:45:00 | 68.83 | 68.83 | 68.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 68.24 | 68.66 | 68.66 | SL hit (close<static) qty=1.00 sl=68.50 alert=retest2 |

### Cycle 138 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 68.50 | 68.63 | 68.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 67.91 | 68.41 | 68.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 68.20 | 68.15 | 68.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 68.20 | 68.15 | 68.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 68.20 | 68.15 | 68.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 68.19 | 68.15 | 68.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 72.00 | 68.92 | 68.67 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 69.02 | 69.93 | 70.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 68.60 | 69.15 | 69.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 65.74 | 65.74 | 66.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:45:00 | 65.63 | 65.74 | 66.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 66.11 | 65.86 | 66.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 66.29 | 65.86 | 66.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 65.80 | 65.85 | 66.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 65.70 | 65.81 | 66.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 65.60 | 65.83 | 66.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 65.69 | 65.83 | 66.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 67.43 | 65.77 | 65.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 67.43 | 65.77 | 65.73 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 65.54 | 65.92 | 65.94 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 66.04 | 65.96 | 65.95 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 65.64 | 65.89 | 65.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 65.00 | 65.67 | 65.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 64.08 | 63.99 | 64.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 64.38 | 63.99 | 64.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 63.69 | 63.93 | 64.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 63.48 | 63.83 | 64.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 62.93 | 62.66 | 62.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 62.93 | 62.66 | 62.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 63.38 | 62.97 | 62.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 65.51 | 65.57 | 64.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 65.51 | 65.57 | 64.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 65.08 | 65.42 | 65.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 65.08 | 65.42 | 65.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 64.82 | 65.30 | 65.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 64.62 | 65.30 | 65.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 64.46 | 65.13 | 64.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 64.46 | 65.13 | 64.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 64.14 | 64.94 | 64.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 64.14 | 64.94 | 64.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 64.21 | 64.79 | 64.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 63.89 | 64.29 | 64.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 61.74 | 61.32 | 61.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 61.74 | 61.32 | 61.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 61.74 | 61.32 | 61.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 62.00 | 61.32 | 61.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 61.59 | 61.37 | 61.85 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 63.43 | 62.11 | 62.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 64.62 | 63.14 | 62.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 64.01 | 64.11 | 63.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 64.01 | 64.11 | 63.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 63.71 | 63.98 | 63.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 64.10 | 63.63 | 63.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:30:00 | 66.15 | 63.84 | 63.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 64.49 | 65.00 | 65.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 64.49 | 65.00 | 65.00 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 65.14 | 65.00 | 65.00 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 64.80 | 64.96 | 64.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 64.55 | 64.88 | 64.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 15:15:00 | 65.03 | 64.85 | 64.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 15:15:00 | 65.03 | 64.85 | 64.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 65.03 | 64.85 | 64.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 66.65 | 64.85 | 64.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 67.15 | 65.31 | 65.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 71.32 | 67.92 | 67.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 70.00 | 70.47 | 69.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:45:00 | 69.99 | 70.47 | 69.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 69.81 | 70.17 | 69.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 69.34 | 70.17 | 69.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 69.10 | 69.97 | 69.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 69.12 | 69.97 | 69.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 68.44 | 69.67 | 69.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 68.44 | 69.67 | 69.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 68.44 | 69.20 | 69.23 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 69.78 | 69.19 | 69.17 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 68.00 | 68.97 | 69.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 67.11 | 68.60 | 68.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 64.95 | 64.86 | 65.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 65.51 | 64.86 | 65.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 65.32 | 64.95 | 65.54 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 68.81 | 66.08 | 65.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 69.40 | 67.58 | 66.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 68.55 | 69.43 | 68.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 68.55 | 69.43 | 68.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 68.55 | 69.43 | 68.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 68.55 | 69.43 | 68.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 68.13 | 69.17 | 68.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 68.13 | 69.17 | 68.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 68.20 | 68.97 | 68.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 68.24 | 68.97 | 68.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 68.29 | 68.74 | 68.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 69.42 | 68.55 | 68.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 69.20 | 70.51 | 70.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 69.20 | 70.51 | 70.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 69.05 | 69.71 | 70.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 68.45 | 67.97 | 68.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 68.45 | 67.97 | 68.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 68.45 | 67.97 | 68.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 68.45 | 67.97 | 68.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 68.37 | 68.10 | 68.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 68.50 | 68.10 | 68.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 68.34 | 68.15 | 68.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 68.53 | 68.15 | 68.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 68.56 | 68.21 | 68.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 69.00 | 68.21 | 68.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 68.30 | 68.23 | 68.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 68.20 | 68.23 | 68.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 68.13 | 68.23 | 68.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 68.14 | 68.21 | 68.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 68.16 | 68.14 | 68.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 68.20 | 68.16 | 68.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 68.35 | 68.16 | 68.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 68.43 | 68.21 | 68.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 68.34 | 68.21 | 68.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 67.85 | 68.14 | 68.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:15:00 | 67.72 | 68.14 | 68.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 67.76 | 68.08 | 68.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 67.79 | 68.02 | 68.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 67.64 | 68.01 | 68.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 67.19 | 67.85 | 68.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 66.95 | 67.85 | 68.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 68.54 | 67.83 | 67.93 | SL hit (close>static) qty=1.00 sl=68.50 alert=retest2 |

### Cycle 157 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 70.16 | 67.61 | 67.41 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 67.82 | 68.67 | 68.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 67.69 | 68.19 | 68.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 65.76 | 65.74 | 66.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 65.76 | 65.74 | 66.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 65.59 | 65.62 | 66.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 66.12 | 65.62 | 66.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 66.44 | 65.78 | 66.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 66.33 | 65.78 | 66.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 66.48 | 65.92 | 66.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 66.48 | 65.92 | 66.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 66.28 | 66.00 | 66.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 66.47 | 66.00 | 66.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 66.17 | 66.13 | 66.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 65.70 | 66.13 | 66.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 66.05 | 66.02 | 66.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 66.15 | 66.04 | 66.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 66.45 | 66.26 | 66.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 66.45 | 66.26 | 66.25 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 66.15 | 66.24 | 66.24 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 66.65 | 66.32 | 66.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 67.25 | 66.66 | 66.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 66.67 | 66.77 | 66.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 66.67 | 66.77 | 66.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 66.67 | 66.77 | 66.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 66.67 | 66.77 | 66.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 66.79 | 66.77 | 66.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:15:00 | 66.76 | 66.77 | 66.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 66.76 | 66.77 | 66.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 65.88 | 66.77 | 66.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 65.90 | 66.60 | 66.57 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 66.01 | 66.48 | 66.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 15:15:00 | 65.50 | 66.00 | 66.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 66.13 | 66.03 | 66.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:15:00 | 66.10 | 66.03 | 66.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 65.95 | 66.01 | 66.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 65.73 | 65.91 | 66.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 15:15:00 | 62.44 | 63.04 | 63.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 61.25 | 60.79 | 61.44 | SL hit (close>ema200) qty=0.50 sl=60.79 alert=retest2 |

### Cycle 163 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 55.20 | 54.89 | 54.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 12:15:00 | 56.16 | 55.46 | 55.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 55.40 | 55.62 | 55.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 55.40 | 55.62 | 55.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 55.40 | 55.62 | 55.36 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 54.80 | 55.17 | 55.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 54.60 | 54.95 | 55.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 54.25 | 54.19 | 54.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 54.25 | 54.19 | 54.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 54.64 | 54.25 | 54.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 54.83 | 54.25 | 54.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 54.45 | 54.29 | 54.43 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 55.31 | 54.62 | 54.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 56.19 | 55.05 | 54.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 57.22 | 57.28 | 56.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 57.45 | 57.28 | 56.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 61.70 | 58.54 | 57.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 63.20 | 58.54 | 57.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-29 09:15:00 | 69.52 | 64.20 | 61.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 65.60 | 66.40 | 66.40 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 69.54 | 66.80 | 66.54 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 66.43 | 67.04 | 67.11 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 68.86 | 67.18 | 67.11 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 65.99 | 67.14 | 67.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 65.20 | 66.31 | 66.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 63.85 | 63.36 | 64.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 64.26 | 63.36 | 64.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 64.03 | 63.49 | 64.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 63.37 | 63.53 | 64.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 63.40 | 63.44 | 63.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 69.34 | 64.61 | 64.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 69.34 | 64.61 | 64.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 70.77 | 67.07 | 65.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 68.40 | 68.83 | 67.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 68.40 | 68.83 | 67.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 68.35 | 68.67 | 67.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 67.80 | 68.67 | 67.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 67.80 | 68.50 | 67.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 67.80 | 68.50 | 67.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 67.90 | 68.38 | 67.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 68.12 | 68.27 | 67.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:15:00 | 68.33 | 68.22 | 67.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 66.80 | 68.05 | 67.89 | SL hit (close<static) qty=1.00 sl=67.63 alert=retest2 |

### Cycle 172 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 66.40 | 67.72 | 67.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 66.32 | 67.44 | 67.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 66.79 | 66.42 | 66.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 66.79 | 66.42 | 66.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 66.79 | 66.42 | 66.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 66.79 | 66.42 | 66.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 66.51 | 66.44 | 66.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:00:00 | 64.76 | 65.73 | 66.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 64.60 | 65.55 | 66.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 64.75 | 65.41 | 66.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 64.75 | 65.41 | 66.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 65.10 | 65.15 | 65.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 65.18 | 65.15 | 65.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 64.66 | 64.25 | 64.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 64.90 | 64.25 | 64.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 64.97 | 64.25 | 64.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 65.57 | 64.25 | 64.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 64.86 | 64.37 | 64.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 68.88 | 65.28 | 65.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 68.88 | 65.28 | 65.00 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 65.70 | 66.78 | 66.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 65.00 | 66.19 | 66.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 63.06 | 62.33 | 63.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 63.06 | 62.33 | 63.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 63.35 | 62.66 | 63.55 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 65.50 | 63.98 | 63.87 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 63.15 | 64.02 | 64.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 62.55 | 63.49 | 63.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 63.20 | 63.13 | 63.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 63.20 | 63.13 | 63.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 64.27 | 63.35 | 63.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:30:00 | 64.44 | 63.35 | 63.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 63.85 | 63.45 | 63.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 64.37 | 63.45 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 64.89 | 63.88 | 63.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 66.10 | 64.33 | 64.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 66.10 | 66.19 | 65.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 65.70 | 66.19 | 65.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 66.60 | 66.28 | 65.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 65.44 | 66.28 | 65.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 65.98 | 66.19 | 65.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 65.75 | 66.19 | 65.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 65.85 | 66.09 | 65.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 65.46 | 66.09 | 65.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 65.60 | 65.99 | 65.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 65.53 | 65.99 | 65.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 65.55 | 65.90 | 65.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:30:00 | 65.50 | 65.90 | 65.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 65.67 | 65.78 | 65.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 65.69 | 65.78 | 65.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 65.38 | 65.70 | 65.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 64.56 | 65.48 | 65.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 64.01 | 63.75 | 64.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 64.01 | 63.75 | 64.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 64.01 | 63.75 | 64.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 64.24 | 63.75 | 64.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 63.49 | 63.61 | 63.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:45:00 | 63.34 | 63.55 | 63.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 63.10 | 63.51 | 63.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 62.03 | 61.81 | 61.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 62.03 | 61.81 | 61.80 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 10:15:00 | 61.57 | 61.79 | 61.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 61.24 | 61.68 | 61.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 09:15:00 | 62.09 | 61.54 | 61.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 62.09 | 61.54 | 61.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 62.09 | 61.54 | 61.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 63.55 | 61.54 | 61.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 10:15:00 | 62.33 | 61.70 | 61.69 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 59.61 | 61.36 | 61.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 10:15:00 | 58.90 | 60.87 | 61.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 56.94 | 56.51 | 57.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 56.94 | 56.51 | 57.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 57.69 | 56.82 | 57.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 57.59 | 56.82 | 57.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 57.12 | 56.88 | 57.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 57.00 | 57.14 | 57.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 54.15 | 56.55 | 57.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 55.27 | 55.04 | 55.84 | SL hit (close>ema200) qty=0.50 sl=55.04 alert=retest2 |

### Cycle 183 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 56.54 | 55.93 | 55.91 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 55.26 | 55.87 | 55.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 54.30 | 55.46 | 55.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 55.90 | 55.47 | 55.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 55.90 | 55.47 | 55.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 55.90 | 55.47 | 55.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 55.90 | 55.47 | 55.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 55.97 | 55.57 | 55.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:30:00 | 56.23 | 55.57 | 55.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 55.64 | 55.69 | 55.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 55.16 | 55.69 | 55.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 52.40 | 53.69 | 54.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 53.16 | 52.75 | 53.47 | SL hit (close>ema200) qty=0.50 sl=52.75 alert=retest2 |

### Cycle 185 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 60.50 | 54.46 | 53.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 62.26 | 57.10 | 55.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 59.68 | 60.57 | 58.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 59.68 | 60.57 | 58.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 59.03 | 60.07 | 59.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 59.03 | 60.07 | 59.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 58.68 | 59.79 | 59.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 56.58 | 59.79 | 59.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 56.15 | 59.06 | 59.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 55.88 | 58.42 | 58.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 56.54 | 56.12 | 57.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 56.54 | 56.12 | 57.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 56.80 | 56.30 | 56.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 56.80 | 56.30 | 56.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 56.84 | 56.41 | 56.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 57.76 | 56.41 | 56.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 57.79 | 56.68 | 57.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 15:15:00 | 56.80 | 57.17 | 57.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 53.96 | 54.98 | 55.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 15:15:00 | 51.12 | 53.32 | 54.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 187 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 55.30 | 54.68 | 54.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 55.80 | 55.00 | 54.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 56.55 | 56.65 | 56.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 58.04 | 56.65 | 56.11 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:15:00 | 60.94 | 59.36 | 58.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 13:15:00 | 59.35 | 59.41 | 58.78 | SL hit (close<ema200) qty=0.50 sl=59.41 alert=retest1 |

### Cycle 188 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 65.64 | 66.37 | 66.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 65.37 | 66.17 | 66.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 66.50 | 65.89 | 66.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 66.50 | 65.89 | 66.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 66.50 | 65.89 | 66.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 67.13 | 65.89 | 66.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 66.82 | 66.07 | 66.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 66.84 | 66.07 | 66.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 66.71 | 66.30 | 66.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 67.34 | 66.65 | 66.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 66.29 | 66.65 | 66.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 66.29 | 66.65 | 66.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 66.29 | 66.65 | 66.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 66.29 | 66.65 | 66.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 66.23 | 66.57 | 66.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 66.23 | 66.57 | 66.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 66.08 | 66.38 | 66.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 65.80 | 66.06 | 66.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 65.63 | 64.98 | 65.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 65.63 | 64.98 | 65.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 65.63 | 64.98 | 65.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 65.63 | 64.98 | 65.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 65.53 | 65.09 | 65.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 65.74 | 65.09 | 65.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 65.29 | 65.13 | 65.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 65.81 | 65.13 | 65.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 64.93 | 65.09 | 65.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 64.72 | 65.09 | 65.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 66.00 | 65.31 | 65.34 | SL hit (close>static) qty=1.00 sl=65.35 alert=retest2 |

### Cycle 191 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 65.70 | 65.42 | 65.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 65.93 | 65.53 | 65.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 11:15:00 | 65.60 | 65.71 | 65.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 11:15:00 | 65.60 | 65.71 | 65.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 65.60 | 65.71 | 65.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:45:00 | 65.56 | 65.71 | 65.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 65.45 | 65.66 | 65.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:45:00 | 65.27 | 65.66 | 65.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 65.49 | 65.63 | 65.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 66.00 | 65.63 | 65.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 11:30:00 | 76.25 | 2024-04-12 12:15:00 | 75.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-04-16 12:15:00 | 72.80 | 2024-04-19 09:15:00 | 69.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-16 12:15:00 | 72.80 | 2024-04-19 12:15:00 | 71.55 | STOP_HIT | 0.50 | 1.72% |
| BUY | retest2 | 2024-04-26 11:45:00 | 78.00 | 2024-04-30 10:15:00 | 74.85 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2024-04-26 13:30:00 | 77.75 | 2024-04-30 10:15:00 | 74.85 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-04-26 14:15:00 | 77.80 | 2024-04-30 10:15:00 | 74.85 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2024-04-26 14:45:00 | 77.55 | 2024-04-30 10:15:00 | 74.85 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2024-05-02 14:15:00 | 74.80 | 2024-05-07 11:15:00 | 71.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 15:00:00 | 74.80 | 2024-05-07 11:15:00 | 71.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 10:30:00 | 74.40 | 2024-05-07 11:15:00 | 70.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 14:15:00 | 74.80 | 2024-05-08 09:15:00 | 71.30 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2024-05-02 15:00:00 | 74.80 | 2024-05-08 09:15:00 | 71.30 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2024-05-03 10:30:00 | 74.40 | 2024-05-08 09:15:00 | 71.30 | STOP_HIT | 0.50 | 4.17% |
| BUY | retest2 | 2024-05-16 15:15:00 | 72.10 | 2024-05-24 10:15:00 | 74.10 | STOP_HIT | 1.00 | 2.77% |
| SELL | retest2 | 2024-05-30 14:30:00 | 70.35 | 2024-06-03 09:15:00 | 73.60 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2024-05-31 09:30:00 | 70.10 | 2024-06-03 09:15:00 | 73.60 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2024-05-31 12:15:00 | 70.20 | 2024-06-03 09:15:00 | 73.60 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2024-05-31 14:30:00 | 70.45 | 2024-06-03 09:15:00 | 73.60 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2024-06-06 12:30:00 | 68.10 | 2024-06-07 10:15:00 | 70.15 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-06-06 14:00:00 | 68.20 | 2024-06-07 10:15:00 | 70.15 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-06-18 09:15:00 | 76.81 | 2024-06-18 09:15:00 | 75.83 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-06-18 11:30:00 | 76.53 | 2024-06-18 13:15:00 | 84.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-18 13:00:00 | 76.51 | 2024-06-18 13:15:00 | 84.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 10:15:00 | 86.34 | 2024-07-08 11:15:00 | 83.89 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-07-22 12:15:00 | 88.15 | 2024-07-23 12:15:00 | 83.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 13:45:00 | 88.38 | 2024-07-23 12:15:00 | 83.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 88.15 | 2024-07-24 09:15:00 | 95.07 | STOP_HIT | 0.50 | -7.85% |
| SELL | retest2 | 2024-07-22 13:45:00 | 88.38 | 2024-07-24 09:15:00 | 95.07 | STOP_HIT | 0.50 | -7.57% |
| BUY | retest2 | 2024-07-31 15:15:00 | 111.60 | 2024-08-01 11:15:00 | 107.10 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-08-06 12:15:00 | 102.20 | 2024-08-12 09:15:00 | 97.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 13:00:00 | 102.25 | 2024-08-12 09:15:00 | 97.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 12:15:00 | 102.20 | 2024-08-12 12:15:00 | 99.17 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2024-08-08 13:00:00 | 102.25 | 2024-08-12 12:15:00 | 99.17 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2024-08-27 14:15:00 | 100.86 | 2024-08-28 09:15:00 | 104.21 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-09-05 10:30:00 | 98.23 | 2024-09-09 09:15:00 | 93.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 13:00:00 | 98.38 | 2024-09-09 09:15:00 | 93.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 14:30:00 | 98.37 | 2024-09-09 09:15:00 | 93.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 15:00:00 | 98.25 | 2024-09-09 09:15:00 | 93.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 10:30:00 | 98.23 | 2024-09-10 09:15:00 | 95.43 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2024-09-05 13:00:00 | 98.38 | 2024-09-10 09:15:00 | 95.43 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2024-09-05 14:30:00 | 98.37 | 2024-09-10 09:15:00 | 95.43 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2024-09-05 15:00:00 | 98.25 | 2024-09-10 09:15:00 | 95.43 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2024-09-09 09:15:00 | 93.01 | 2024-09-13 11:15:00 | 95.83 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-09-11 12:45:00 | 95.08 | 2024-09-13 11:15:00 | 95.83 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-30 13:15:00 | 89.18 | 2024-10-04 09:15:00 | 84.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 88.83 | 2024-10-04 09:15:00 | 84.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:15:00 | 89.18 | 2024-10-08 09:15:00 | 80.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 88.83 | 2024-10-08 09:15:00 | 83.50 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest2 | 2024-10-17 09:45:00 | 82.89 | 2024-10-22 09:15:00 | 78.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 13:45:00 | 82.79 | 2024-10-22 09:15:00 | 78.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:45:00 | 82.89 | 2024-10-23 09:15:00 | 74.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 13:45:00 | 82.79 | 2024-10-23 09:15:00 | 74.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-19 12:30:00 | 76.71 | 2024-11-25 10:15:00 | 76.46 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-11-19 13:00:00 | 76.69 | 2024-11-25 10:15:00 | 76.46 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-11-19 13:30:00 | 76.67 | 2024-11-25 10:15:00 | 76.46 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-11-25 09:45:00 | 76.28 | 2024-11-25 10:15:00 | 76.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-11-28 09:15:00 | 80.04 | 2024-12-10 12:15:00 | 81.89 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2024-11-29 09:45:00 | 79.10 | 2024-12-10 12:15:00 | 81.89 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2024-11-29 14:00:00 | 78.69 | 2024-12-10 12:15:00 | 81.89 | STOP_HIT | 1.00 | 4.07% |
| BUY | retest2 | 2024-12-02 11:00:00 | 79.76 | 2024-12-10 12:15:00 | 81.89 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2024-12-05 09:15:00 | 81.87 | 2024-12-10 12:15:00 | 81.89 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-12-06 09:30:00 | 82.09 | 2024-12-10 12:15:00 | 81.89 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-12-16 12:00:00 | 79.19 | 2024-12-19 09:15:00 | 75.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:00:00 | 79.19 | 2024-12-26 09:15:00 | 73.18 | STOP_HIT | 0.50 | 7.59% |
| BUY | retest2 | 2025-01-02 13:30:00 | 73.98 | 2025-01-06 09:15:00 | 73.06 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest1 | 2025-01-07 15:15:00 | 70.50 | 2025-01-08 10:15:00 | 77.64 | STOP_HIT | 1.00 | -10.13% |
| SELL | retest2 | 2025-01-27 09:15:00 | 67.10 | 2025-01-28 10:15:00 | 63.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 67.10 | 2025-01-28 12:15:00 | 66.02 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-02-04 10:15:00 | 66.79 | 2025-02-05 09:15:00 | 68.46 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-03-04 12:15:00 | 49.87 | 2025-03-05 09:15:00 | 51.34 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-03-18 12:30:00 | 50.71 | 2025-03-19 09:15:00 | 52.50 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-03-26 12:15:00 | 53.06 | 2025-03-27 13:15:00 | 55.85 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2025-03-26 13:00:00 | 52.90 | 2025-03-27 13:15:00 | 55.85 | STOP_HIT | 1.00 | -5.58% |
| SELL | retest2 | 2025-04-01 11:30:00 | 53.41 | 2025-04-01 13:15:00 | 53.89 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-04-02 12:00:00 | 55.18 | 2025-04-04 10:15:00 | 52.75 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2025-04-02 14:30:00 | 54.61 | 2025-04-04 10:15:00 | 52.75 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-04-03 09:15:00 | 54.61 | 2025-04-04 10:15:00 | 52.75 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-04-03 10:30:00 | 54.56 | 2025-04-04 10:15:00 | 52.75 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-04-08 10:30:00 | 49.48 | 2025-04-11 12:15:00 | 50.45 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-04-09 09:15:00 | 48.88 | 2025-04-11 12:15:00 | 50.45 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-04-24 11:15:00 | 62.12 | 2025-04-28 14:15:00 | 58.09 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2025-05-06 09:30:00 | 55.75 | 2025-05-07 09:15:00 | 52.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:30:00 | 55.75 | 2025-05-07 15:15:00 | 54.10 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-05-12 10:15:00 | 55.75 | 2025-05-12 10:15:00 | 56.04 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-05-23 13:15:00 | 65.75 | 2025-05-28 14:15:00 | 72.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-05 09:15:00 | 76.84 | 2025-06-06 12:15:00 | 73.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 09:15:00 | 76.84 | 2025-06-09 09:15:00 | 74.50 | STOP_HIT | 0.50 | 3.05% |
| BUY | retest2 | 2025-06-25 09:15:00 | 70.11 | 2025-07-02 12:15:00 | 71.20 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-07-04 11:30:00 | 70.31 | 2025-07-15 15:15:00 | 68.60 | STOP_HIT | 1.00 | 2.43% |
| SELL | retest2 | 2025-07-07 09:45:00 | 70.18 | 2025-07-15 15:15:00 | 68.60 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2025-07-08 09:30:00 | 70.24 | 2025-07-15 15:15:00 | 68.60 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-07-17 09:15:00 | 69.00 | 2025-07-17 14:15:00 | 68.24 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-17 12:45:00 | 68.83 | 2025-07-17 14:15:00 | 68.24 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-30 10:45:00 | 65.70 | 2025-08-01 11:15:00 | 67.43 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-07-30 13:45:00 | 65.60 | 2025-08-01 11:15:00 | 67.43 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-07-30 14:15:00 | 65.69 | 2025-08-01 11:15:00 | 67.43 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-08-08 10:45:00 | 63.48 | 2025-08-18 11:15:00 | 62.93 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-09-05 09:45:00 | 64.10 | 2025-09-11 13:15:00 | 64.49 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-09-05 10:30:00 | 66.15 | 2025-09-11 13:15:00 | 64.49 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-10-08 09:15:00 | 69.42 | 2025-10-13 10:15:00 | 69.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-10-16 11:15:00 | 68.20 | 2025-10-21 13:15:00 | 68.54 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-10-16 12:30:00 | 68.13 | 2025-10-21 13:15:00 | 68.54 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-10-16 14:00:00 | 68.14 | 2025-10-21 13:15:00 | 68.54 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-17 10:00:00 | 68.16 | 2025-10-21 13:15:00 | 68.54 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-17 13:15:00 | 67.72 | 2025-10-21 13:15:00 | 68.54 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-17 14:15:00 | 67.76 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-10-17 15:00:00 | 67.79 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-10-20 09:15:00 | 67.64 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-10-20 10:15:00 | 66.95 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2025-10-24 10:15:00 | 67.07 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-10-24 10:45:00 | 67.08 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-10-27 10:45:00 | 67.06 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2025-10-28 12:00:00 | 66.96 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2025-10-28 14:00:00 | 66.83 | 2025-10-29 09:15:00 | 70.16 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-11-10 15:15:00 | 65.70 | 2025-11-11 14:15:00 | 66.45 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-11 09:45:00 | 66.05 | 2025-11-11 14:15:00 | 66.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-11 12:00:00 | 66.15 | 2025-11-11 14:15:00 | 66.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-17 11:30:00 | 65.73 | 2025-11-21 15:15:00 | 62.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 65.73 | 2025-11-26 09:15:00 | 61.25 | STOP_HIT | 0.50 | 6.82% |
| BUY | retest2 | 2025-12-26 10:15:00 | 63.20 | 2025-12-29 09:15:00 | 69.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 63.37 | 2026-01-14 09:15:00 | 69.34 | STOP_HIT | 1.00 | -9.42% |
| SELL | retest2 | 2026-01-13 15:15:00 | 63.40 | 2026-01-14 09:15:00 | 69.34 | STOP_HIT | 1.00 | -9.37% |
| BUY | retest2 | 2026-01-19 13:15:00 | 68.12 | 2026-01-20 09:15:00 | 66.80 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-01-19 14:15:00 | 68.33 | 2026-01-20 09:15:00 | 66.80 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-01-22 11:00:00 | 64.76 | 2026-01-28 10:15:00 | 68.88 | STOP_HIT | 1.00 | -6.36% |
| SELL | retest2 | 2026-01-22 11:30:00 | 64.60 | 2026-01-28 10:15:00 | 68.88 | STOP_HIT | 1.00 | -6.63% |
| SELL | retest2 | 2026-01-22 12:45:00 | 64.75 | 2026-01-28 10:15:00 | 68.88 | STOP_HIT | 1.00 | -6.38% |
| SELL | retest2 | 2026-01-22 13:15:00 | 64.75 | 2026-01-28 10:15:00 | 68.88 | STOP_HIT | 1.00 | -6.38% |
| SELL | retest2 | 2026-02-18 12:45:00 | 63.34 | 2026-02-25 14:15:00 | 62.03 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2026-02-18 15:15:00 | 63.10 | 2026-02-25 14:15:00 | 62.03 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2026-03-06 15:15:00 | 57.00 | 2026-03-09 09:15:00 | 54.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 57.00 | 2026-03-10 09:15:00 | 55.27 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-03-13 09:15:00 | 55.16 | 2026-03-16 09:15:00 | 52.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 55.16 | 2026-03-17 09:15:00 | 53.16 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2026-03-25 15:15:00 | 56.80 | 2026-03-30 09:15:00 | 53.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 15:15:00 | 56.80 | 2026-03-30 15:15:00 | 51.12 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 58.04 | 2026-04-10 11:15:00 | 60.94 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 58.04 | 2026-04-10 13:15:00 | 59.35 | STOP_HIT | 0.50 | 2.26% |
| BUY | retest2 | 2026-04-13 10:15:00 | 58.55 | 2026-04-17 10:15:00 | 64.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 13:15:00 | 64.72 | 2026-05-05 09:15:00 | 66.00 | STOP_HIT | 1.00 | -1.98% |
