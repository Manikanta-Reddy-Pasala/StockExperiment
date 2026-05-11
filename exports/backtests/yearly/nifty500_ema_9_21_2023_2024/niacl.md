# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 163.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 227 |
| ALERT1 | 156 |
| ALERT2 | 157 |
| ALERT2_SKIP | 109 |
| ALERT3 | 305 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 110 |
| PARTIAL | 30 |
| TARGET_HIT | 16 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 77 / 62
- **Target hits / Stop hits / Partials:** 16 / 93 / 30
- **Avg / median % per leg:** 2.11% / 0.50%
- **Sum % (uncompounded):** 293.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 11 | 35.5% | 8 | 23 | 0 | 1.69% | 52.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.18% | -3.2% |
| BUY @ 3rd Alert (retest2) | 30 | 11 | 36.7% | 8 | 22 | 0 | 1.86% | 55.7% |
| SELL (all) | 108 | 66 | 61.1% | 8 | 70 | 30 | 2.23% | 240.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.13% | -1.1% |
| SELL @ 3rd Alert (retest2) | 107 | 66 | 61.7% | 8 | 69 | 30 | 2.26% | 242.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.15% | -4.3% |
| retest2 (combined) | 137 | 77 | 56.2% | 16 | 91 | 30 | 2.17% | 297.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 123.40 | 117.40 | 116.75 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 10:15:00 | 117.30 | 119.05 | 119.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 15:15:00 | 116.00 | 117.05 | 117.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 10:15:00 | 117.50 | 117.12 | 117.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 10:15:00 | 117.50 | 117.12 | 117.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 117.50 | 117.12 | 117.68 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 13:15:00 | 119.20 | 117.46 | 117.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 121.45 | 119.32 | 118.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 15:15:00 | 119.65 | 119.74 | 119.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 15:15:00 | 119.65 | 119.74 | 119.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 119.65 | 119.74 | 119.19 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 13:15:00 | 119.10 | 119.15 | 119.15 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 14:15:00 | 119.50 | 119.22 | 119.18 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 15:15:00 | 118.85 | 119.14 | 119.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 11:15:00 | 116.80 | 118.41 | 118.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 13:15:00 | 121.00 | 118.66 | 118.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 13:15:00 | 121.00 | 118.66 | 118.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 121.00 | 118.66 | 118.83 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 119.60 | 118.94 | 118.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 11:15:00 | 120.50 | 119.42 | 119.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 14:15:00 | 119.65 | 119.74 | 119.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 119.65 | 119.72 | 119.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 119.65 | 119.72 | 119.41 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 15:15:00 | 118.80 | 119.45 | 119.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 09:15:00 | 118.70 | 119.30 | 119.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 09:15:00 | 118.85 | 118.50 | 118.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 118.85 | 118.50 | 118.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 118.85 | 118.50 | 118.84 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 120.45 | 118.98 | 118.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 13:15:00 | 121.45 | 119.94 | 119.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 14:15:00 | 120.85 | 121.72 | 120.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 14:15:00 | 120.85 | 121.72 | 120.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 120.85 | 121.72 | 120.84 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 117.95 | 120.35 | 120.39 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 13:15:00 | 121.05 | 119.18 | 119.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 12:15:00 | 122.15 | 120.56 | 120.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 13:15:00 | 122.40 | 122.75 | 121.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 13:15:00 | 122.40 | 122.75 | 121.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 122.40 | 122.75 | 121.72 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 120.45 | 121.93 | 122.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 119.05 | 120.86 | 121.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 10:15:00 | 116.50 | 115.86 | 117.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 116.65 | 116.15 | 117.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 116.65 | 116.15 | 117.09 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 14:15:00 | 118.40 | 117.34 | 117.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 119.25 | 117.91 | 117.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 12:15:00 | 118.00 | 118.05 | 117.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 13:15:00 | 117.60 | 117.96 | 117.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 117.60 | 117.96 | 117.72 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 117.50 | 117.57 | 117.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 12:15:00 | 117.15 | 117.45 | 117.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 118.50 | 117.44 | 117.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 118.50 | 117.44 | 117.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 118.50 | 117.44 | 117.47 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 118.35 | 117.62 | 117.55 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 10:15:00 | 117.35 | 117.63 | 117.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 116.50 | 117.31 | 117.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 12:15:00 | 116.50 | 116.50 | 116.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 09:15:00 | 118.55 | 116.79 | 116.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 118.55 | 116.79 | 116.89 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 118.50 | 117.13 | 117.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 09:15:00 | 119.65 | 118.45 | 117.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 11:15:00 | 118.35 | 118.49 | 117.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 12:15:00 | 118.05 | 118.40 | 117.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 118.05 | 118.40 | 117.97 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 12:15:00 | 118.00 | 119.25 | 119.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 116.50 | 118.14 | 118.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 117.35 | 117.32 | 117.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 12:15:00 | 117.35 | 117.36 | 117.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 117.35 | 117.36 | 117.71 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 119.00 | 116.93 | 116.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 11:15:00 | 120.30 | 117.91 | 117.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 14:15:00 | 123.60 | 123.73 | 122.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 122.50 | 123.48 | 122.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 122.50 | 123.48 | 122.45 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 15:15:00 | 122.00 | 122.53 | 122.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 121.05 | 122.24 | 122.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 11:15:00 | 122.40 | 122.14 | 122.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 11:15:00 | 122.40 | 122.14 | 122.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 122.40 | 122.14 | 122.33 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 14:15:00 | 124.25 | 122.31 | 122.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 128.65 | 123.93 | 122.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 15:15:00 | 126.10 | 126.32 | 124.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 11:15:00 | 125.90 | 126.47 | 125.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 125.90 | 126.47 | 125.34 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 123.30 | 124.63 | 124.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 13:15:00 | 121.65 | 123.69 | 124.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 126.90 | 123.92 | 124.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 126.90 | 123.92 | 124.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 126.90 | 123.92 | 124.18 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 126.90 | 124.51 | 124.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 127.70 | 126.11 | 125.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 13:15:00 | 127.70 | 127.76 | 127.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 14:15:00 | 127.05 | 127.62 | 127.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 127.05 | 127.62 | 127.03 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 13:15:00 | 126.25 | 126.77 | 126.81 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 127.10 | 126.84 | 126.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 15:15:00 | 127.30 | 126.93 | 126.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 126.60 | 126.91 | 126.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 10:15:00 | 126.60 | 126.91 | 126.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 126.60 | 126.91 | 126.88 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 126.10 | 126.71 | 126.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 15:15:00 | 125.25 | 126.16 | 126.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 125.20 | 125.10 | 125.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 124.05 | 123.79 | 124.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 124.05 | 123.79 | 124.48 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 15:15:00 | 124.65 | 124.57 | 124.57 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 09:15:00 | 123.80 | 124.42 | 124.50 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 129.50 | 125.10 | 124.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 133.40 | 126.76 | 125.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 131.15 | 131.43 | 129.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 14:15:00 | 131.20 | 131.77 | 130.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 131.20 | 131.77 | 130.82 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 128.25 | 130.04 | 130.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 12:15:00 | 128.00 | 129.63 | 130.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 131.80 | 129.59 | 129.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 131.80 | 129.59 | 129.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 131.80 | 129.59 | 129.81 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 130.90 | 130.08 | 130.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 131.80 | 130.83 | 130.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 10:15:00 | 130.70 | 130.80 | 130.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 11:15:00 | 130.75 | 130.79 | 130.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 130.75 | 130.79 | 130.49 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 15:15:00 | 129.75 | 130.36 | 130.38 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 10:15:00 | 130.80 | 130.46 | 130.42 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 12:15:00 | 130.05 | 130.38 | 130.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 13:15:00 | 129.75 | 130.25 | 130.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 14:15:00 | 130.60 | 130.32 | 130.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 130.60 | 130.32 | 130.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 130.60 | 130.32 | 130.36 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 15:15:00 | 130.25 | 129.97 | 129.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 133.90 | 130.75 | 130.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 133.70 | 133.76 | 132.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 133.70 | 133.76 | 132.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 133.70 | 133.76 | 132.60 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 13:15:00 | 132.95 | 134.10 | 134.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 128.25 | 132.64 | 133.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 127.40 | 127.06 | 129.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 12:15:00 | 128.10 | 127.26 | 129.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 128.10 | 127.26 | 129.15 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 134.40 | 130.31 | 129.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 141.75 | 133.98 | 131.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 15:15:00 | 142.60 | 142.62 | 139.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 11:15:00 | 140.20 | 143.16 | 142.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 140.20 | 143.16 | 142.04 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 139.10 | 141.40 | 141.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 137.40 | 140.28 | 140.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 139.55 | 139.31 | 140.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 139.55 | 139.31 | 140.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 139.55 | 139.31 | 140.02 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 141.80 | 139.33 | 139.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 12:15:00 | 142.70 | 140.60 | 139.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 15:15:00 | 140.60 | 140.89 | 140.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 15:15:00 | 140.60 | 140.89 | 140.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 140.60 | 140.89 | 140.17 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 10:15:00 | 138.90 | 139.85 | 139.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 136.95 | 138.83 | 139.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 136.45 | 136.18 | 137.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 15:15:00 | 137.40 | 136.47 | 137.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 137.40 | 136.47 | 137.10 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 139.55 | 137.64 | 137.55 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 135.85 | 137.59 | 137.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 134.85 | 137.04 | 137.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 136.60 | 136.31 | 136.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 136.60 | 136.31 | 136.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 136.60 | 136.31 | 136.95 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 138.10 | 137.33 | 137.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 139.20 | 137.80 | 137.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 137.25 | 138.07 | 137.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 13:15:00 | 137.25 | 138.07 | 137.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 137.25 | 138.07 | 137.77 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 09:15:00 | 137.10 | 137.58 | 137.60 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 13:15:00 | 137.80 | 137.51 | 137.48 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 137.10 | 137.43 | 137.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 136.75 | 137.29 | 137.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 137.80 | 137.39 | 137.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 137.80 | 137.39 | 137.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 137.80 | 137.39 | 137.42 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 10:15:00 | 139.75 | 137.87 | 137.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 141.60 | 139.42 | 138.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 09:15:00 | 142.15 | 142.36 | 141.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 139.50 | 141.79 | 140.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 139.50 | 141.79 | 140.93 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 13:15:00 | 142.30 | 145.32 | 145.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 138.15 | 143.89 | 144.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 135.60 | 134.86 | 137.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 135.60 | 134.86 | 137.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 135.60 | 134.86 | 137.65 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 141.25 | 137.24 | 137.09 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 136.90 | 137.86 | 137.90 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 139.30 | 138.05 | 137.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 140.35 | 138.78 | 138.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 138.50 | 138.82 | 138.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 14:15:00 | 138.50 | 138.82 | 138.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 138.50 | 138.82 | 138.39 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 10:15:00 | 137.30 | 141.94 | 142.13 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 10:15:00 | 141.20 | 139.70 | 139.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 10:15:00 | 142.90 | 141.12 | 140.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 150.20 | 150.87 | 148.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 10:15:00 | 235.40 | 243.54 | 234.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 235.40 | 243.54 | 234.81 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 15:15:00 | 232.15 | 236.19 | 236.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 10:15:00 | 226.40 | 233.49 | 235.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 232.30 | 229.76 | 232.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 232.30 | 229.76 | 232.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 232.30 | 229.76 | 232.14 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 240.55 | 231.55 | 231.26 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 11:15:00 | 230.95 | 233.04 | 233.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 12:15:00 | 229.90 | 232.42 | 232.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 12:15:00 | 226.35 | 226.16 | 228.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 12:15:00 | 226.35 | 226.16 | 228.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 226.35 | 226.16 | 228.16 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 231.25 | 228.00 | 227.65 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 216.20 | 225.86 | 226.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 213.90 | 223.47 | 225.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 220.15 | 217.56 | 220.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 220.15 | 217.56 | 220.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 220.15 | 217.56 | 220.17 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 13:15:00 | 218.00 | 214.01 | 213.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 13:15:00 | 219.55 | 216.65 | 215.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 225.55 | 227.76 | 225.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 225.55 | 227.76 | 225.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 225.55 | 227.76 | 225.14 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 221.10 | 223.90 | 224.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 15:15:00 | 220.25 | 222.06 | 222.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 216.70 | 214.56 | 216.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 216.70 | 214.56 | 216.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 216.70 | 214.56 | 216.33 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 09:15:00 | 218.15 | 216.00 | 215.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 10:15:00 | 220.75 | 216.95 | 216.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 14:15:00 | 223.10 | 225.72 | 222.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 14:15:00 | 223.10 | 225.72 | 222.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 223.10 | 225.72 | 222.84 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 226.95 | 233.71 | 234.42 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 236.20 | 233.72 | 233.52 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 12:15:00 | 232.60 | 233.36 | 233.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 13:15:00 | 231.40 | 232.97 | 233.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 235.75 | 232.95 | 233.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 235.75 | 232.95 | 233.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 235.75 | 232.95 | 233.08 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 235.15 | 233.39 | 233.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 240.30 | 234.77 | 233.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 12:15:00 | 242.75 | 245.84 | 242.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 12:15:00 | 242.75 | 245.84 | 242.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 242.75 | 245.84 | 242.50 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 14:15:00 | 240.15 | 241.76 | 241.94 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 244.80 | 242.09 | 242.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 263.65 | 246.57 | 244.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 14:15:00 | 272.45 | 274.61 | 269.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 293.50 | 297.20 | 287.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 293.50 | 297.20 | 287.31 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 268.70 | 283.97 | 285.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 12:15:00 | 257.25 | 275.06 | 280.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 249.40 | 246.63 | 257.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 11:15:00 | 249.00 | 247.41 | 255.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 249.00 | 247.41 | 255.74 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 290.55 | 257.74 | 254.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 292.30 | 275.88 | 265.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 286.65 | 290.05 | 279.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 10:15:00 | 281.90 | 288.42 | 279.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 281.90 | 288.42 | 279.69 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 09:15:00 | 280.00 | 281.92 | 282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 13:15:00 | 278.05 | 280.56 | 281.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 276.50 | 271.69 | 275.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 276.50 | 271.69 | 275.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 276.50 | 271.69 | 275.08 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 15:15:00 | 268.25 | 264.52 | 264.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 272.00 | 266.57 | 265.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 265.20 | 267.06 | 265.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 265.20 | 267.06 | 265.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 265.20 | 267.06 | 265.68 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 261.75 | 268.39 | 268.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 259.30 | 266.57 | 267.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 227.40 | 226.96 | 236.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 232.40 | 228.05 | 235.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 232.40 | 228.05 | 235.74 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 229.35 | 227.89 | 227.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 231.75 | 228.66 | 228.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 231.40 | 232.11 | 230.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 231.40 | 232.11 | 230.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 231.40 | 232.11 | 230.80 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 226.35 | 230.12 | 230.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 15:15:00 | 224.10 | 228.92 | 229.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 228.90 | 228.81 | 229.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 10:15:00 | 228.90 | 228.81 | 229.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 228.90 | 228.81 | 229.55 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 13:15:00 | 236.40 | 230.73 | 230.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 09:15:00 | 237.50 | 231.85 | 230.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 12:15:00 | 230.65 | 232.01 | 231.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 12:15:00 | 230.65 | 232.01 | 231.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 230.65 | 232.01 | 231.24 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 15:15:00 | 228.80 | 230.50 | 230.68 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 233.55 | 231.11 | 230.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 237.25 | 233.18 | 232.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 10:15:00 | 239.80 | 239.89 | 237.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 13:15:00 | 237.10 | 239.35 | 237.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 237.10 | 239.35 | 237.97 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 237.05 | 237.80 | 237.89 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 239.60 | 238.16 | 238.05 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 236.80 | 238.03 | 238.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 234.95 | 237.00 | 237.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 236.65 | 236.42 | 237.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 236.65 | 236.42 | 237.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 236.65 | 236.42 | 237.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 228.45 | 230.39 | 232.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 217.03 | 225.46 | 228.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 221.50 | 221.26 | 224.39 | SL hit (close>ema200) qty=0.50 sl=221.26 alert=retest2 |

### Cycle 81 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 224.25 | 221.85 | 221.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 12:15:00 | 226.85 | 224.61 | 223.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 225.05 | 225.27 | 224.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 15:15:00 | 225.05 | 225.27 | 224.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 225.05 | 225.27 | 224.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:15:00 | 227.40 | 225.30 | 224.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:15:00 | 227.45 | 225.60 | 224.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 15:15:00 | 227.30 | 226.63 | 225.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:15:00 | 228.50 | 226.62 | 225.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-29 09:15:00 | 250.14 | 239.39 | 232.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 238.95 | 243.23 | 243.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 229.70 | 239.51 | 241.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 224.45 | 223.50 | 227.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:45:00 | 224.15 | 223.50 | 227.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 219.35 | 218.57 | 220.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 220.75 | 218.57 | 220.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 220.15 | 218.89 | 220.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 220.15 | 218.89 | 220.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 218.15 | 218.97 | 220.32 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 226.00 | 221.00 | 220.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 227.10 | 223.57 | 222.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 232.50 | 232.53 | 229.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 232.50 | 232.53 | 229.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 232.75 | 233.05 | 231.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 232.75 | 233.05 | 231.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 235.70 | 234.13 | 232.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 235.80 | 234.13 | 232.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 233.80 | 234.16 | 233.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:30:00 | 233.40 | 234.16 | 233.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 232.75 | 233.88 | 233.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 232.75 | 233.88 | 233.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 231.20 | 233.34 | 232.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:30:00 | 230.70 | 233.34 | 232.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 231.75 | 232.90 | 232.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 240.45 | 232.90 | 232.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 239.50 | 241.92 | 242.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 239.50 | 241.92 | 242.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 15:15:00 | 238.50 | 241.24 | 241.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 232.50 | 231.40 | 233.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 232.50 | 231.40 | 233.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 232.50 | 231.40 | 233.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:30:00 | 232.40 | 231.40 | 233.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 234.90 | 232.10 | 234.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:45:00 | 230.35 | 232.86 | 233.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:30:00 | 229.25 | 230.13 | 231.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 234.65 | 231.55 | 231.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 234.65 | 231.55 | 231.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 238.65 | 233.80 | 232.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 229.10 | 234.60 | 233.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 229.10 | 234.60 | 233.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 229.10 | 234.60 | 233.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 227.40 | 234.60 | 233.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 216.50 | 230.98 | 231.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 200.25 | 224.84 | 229.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 209.40 | 208.63 | 215.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 209.40 | 208.63 | 215.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 219.40 | 211.02 | 214.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 220.05 | 211.02 | 214.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 220.85 | 212.99 | 215.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 221.25 | 212.99 | 215.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 218.05 | 215.18 | 215.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 218.40 | 215.18 | 215.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 217.55 | 216.07 | 216.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 219.40 | 216.07 | 216.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 218.90 | 216.64 | 216.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 221.50 | 219.20 | 217.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 239.91 | 240.22 | 236.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 14:00:00 | 239.91 | 240.22 | 236.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 243.29 | 244.98 | 243.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 242.55 | 244.98 | 243.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 240.95 | 244.18 | 242.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 240.92 | 244.18 | 242.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 241.24 | 243.59 | 242.73 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 240.30 | 242.24 | 242.27 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 243.00 | 242.40 | 242.34 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 237.96 | 241.56 | 241.97 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 247.24 | 242.18 | 241.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 15:15:00 | 254.00 | 247.07 | 244.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 252.88 | 253.85 | 249.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 252.88 | 253.85 | 249.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 253.08 | 253.30 | 250.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 248.76 | 253.30 | 250.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 249.32 | 252.51 | 250.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 249.32 | 252.51 | 250.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 249.39 | 251.88 | 250.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 248.80 | 251.88 | 250.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 245.00 | 248.40 | 248.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 244.55 | 247.63 | 248.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 14:15:00 | 240.90 | 240.23 | 242.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 15:00:00 | 240.90 | 240.23 | 242.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 245.19 | 241.30 | 242.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 245.00 | 241.30 | 242.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 241.07 | 241.25 | 242.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:30:00 | 240.19 | 240.61 | 241.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 10:15:00 | 246.25 | 239.54 | 239.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 246.25 | 239.54 | 239.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 249.56 | 241.55 | 240.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 244.14 | 244.25 | 242.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:45:00 | 244.25 | 244.25 | 242.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 242.32 | 243.78 | 242.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 242.32 | 243.78 | 242.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 240.39 | 243.10 | 242.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 240.39 | 243.10 | 242.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 241.23 | 242.73 | 242.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:15:00 | 242.72 | 242.54 | 242.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 242.07 | 242.41 | 242.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:00:00 | 241.88 | 242.31 | 242.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 13:30:00 | 242.10 | 242.90 | 242.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 243.10 | 243.63 | 242.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 243.69 | 243.63 | 242.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 248.69 | 244.64 | 243.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 242.99 | 244.64 | 243.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2024-07-05 09:15:00 | 266.99 | 254.71 | 249.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 281.05 | 285.41 | 286.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 280.23 | 283.77 | 285.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 270.90 | 270.12 | 274.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:00:00 | 270.90 | 270.12 | 274.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 264.50 | 260.85 | 265.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 264.50 | 260.85 | 265.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 259.09 | 259.79 | 262.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:30:00 | 265.70 | 259.79 | 262.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 263.72 | 260.57 | 262.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 263.72 | 260.57 | 262.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 259.70 | 260.40 | 262.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 13:30:00 | 258.82 | 259.79 | 261.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:30:00 | 258.43 | 259.74 | 261.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 258.87 | 259.74 | 261.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:00:00 | 258.99 | 259.59 | 261.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 11:15:00 | 279.89 | 263.65 | 262.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 279.89 | 263.65 | 262.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 294.44 | 269.81 | 265.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 278.82 | 283.49 | 276.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 278.82 | 283.49 | 276.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 280.00 | 282.79 | 277.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 282.61 | 281.94 | 277.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:45:00 | 282.28 | 281.93 | 278.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 15:15:00 | 280.25 | 286.36 | 286.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 280.25 | 286.36 | 286.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 278.20 | 284.73 | 286.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 268.15 | 266.60 | 272.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 261.55 | 264.35 | 269.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 262.65 | 260.41 | 263.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 262.65 | 260.41 | 263.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 264.50 | 261.22 | 263.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 264.50 | 261.22 | 263.99 | SL hit (close>ema400) qty=1.00 sl=263.99 alert=retest1 |

### Cycle 97 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 249.70 | 242.99 | 242.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 254.50 | 249.47 | 246.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 268.80 | 269.16 | 262.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 11:15:00 | 264.70 | 267.50 | 265.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 264.70 | 267.50 | 265.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 264.70 | 267.50 | 265.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 264.90 | 266.98 | 265.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 264.90 | 266.98 | 265.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 264.25 | 266.14 | 265.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:30:00 | 263.85 | 266.14 | 265.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 264.30 | 265.77 | 265.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 263.50 | 265.77 | 265.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 264.05 | 265.26 | 265.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 264.05 | 265.26 | 265.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 264.85 | 265.18 | 265.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 268.00 | 264.95 | 264.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 262.35 | 267.62 | 268.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 262.35 | 267.62 | 268.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 14:15:00 | 261.85 | 265.24 | 266.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 264.70 | 264.61 | 266.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 264.70 | 264.61 | 266.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 264.70 | 264.61 | 266.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 262.60 | 264.20 | 265.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:00:00 | 262.25 | 264.20 | 265.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:00:00 | 260.20 | 263.40 | 264.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 12:15:00 | 268.35 | 264.54 | 265.11 | SL hit (close>static) qty=1.00 sl=266.80 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 266.45 | 265.36 | 265.23 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 263.70 | 265.06 | 265.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 11:15:00 | 263.45 | 264.74 | 264.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 263.40 | 263.39 | 264.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 263.40 | 263.39 | 264.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 263.40 | 263.39 | 264.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 263.05 | 263.39 | 264.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 264.70 | 263.65 | 264.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 264.70 | 263.65 | 264.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 265.70 | 264.06 | 264.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:45:00 | 265.65 | 264.06 | 264.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 264.60 | 264.32 | 264.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:30:00 | 265.20 | 264.32 | 264.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 263.85 | 264.22 | 264.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 263.85 | 264.22 | 264.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 263.50 | 264.08 | 264.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 263.30 | 264.08 | 264.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 256.25 | 262.51 | 263.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:30:00 | 255.65 | 261.14 | 262.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 11:00:00 | 255.65 | 261.14 | 262.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:15:00 | 254.80 | 258.84 | 261.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 253.40 | 255.62 | 257.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 250.20 | 254.54 | 256.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:00:00 | 248.25 | 253.28 | 255.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:15:00 | 242.87 | 246.20 | 249.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:15:00 | 242.87 | 246.20 | 249.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:15:00 | 242.06 | 246.20 | 249.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:15:00 | 240.73 | 246.20 | 249.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 244.40 | 243.68 | 247.53 | SL hit (close>ema200) qty=0.50 sl=243.68 alert=retest2 |

### Cycle 101 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 236.45 | 235.11 | 235.02 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 234.95 | 235.32 | 235.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 232.05 | 234.44 | 234.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 11:15:00 | 239.00 | 234.98 | 235.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 239.00 | 234.98 | 235.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 239.00 | 234.98 | 235.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 239.00 | 234.98 | 235.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 12:15:00 | 238.75 | 235.74 | 235.39 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 234.10 | 235.71 | 235.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 233.15 | 234.97 | 235.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 231.89 | 231.79 | 233.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 231.89 | 231.79 | 233.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 231.89 | 231.79 | 233.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 230.10 | 232.25 | 232.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:45:00 | 230.11 | 231.75 | 232.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 218.59 | 222.68 | 226.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 218.60 | 222.68 | 226.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 13:15:00 | 207.09 | 214.34 | 220.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 217.26 | 212.87 | 212.47 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 212.40 | 213.94 | 214.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 211.00 | 213.01 | 213.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 208.99 | 208.98 | 210.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:30:00 | 208.84 | 208.98 | 210.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 211.64 | 209.58 | 210.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 209.30 | 209.68 | 210.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 198.84 | 203.60 | 206.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 199.94 | 199.68 | 203.17 | SL hit (close>ema200) qty=0.50 sl=199.68 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 196.00 | 194.63 | 194.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 200.44 | 195.79 | 195.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 199.30 | 199.39 | 197.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 15:15:00 | 199.30 | 199.39 | 197.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 199.30 | 199.39 | 197.61 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 195.38 | 197.06 | 197.14 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 200.61 | 197.77 | 197.45 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 193.07 | 197.11 | 197.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 190.08 | 195.70 | 196.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 190.73 | 190.15 | 191.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 10:00:00 | 190.73 | 190.15 | 191.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 191.25 | 190.37 | 191.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:45:00 | 191.78 | 190.37 | 191.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 190.20 | 190.34 | 191.43 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 194.15 | 192.30 | 192.10 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 189.90 | 192.23 | 192.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 188.41 | 191.46 | 192.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 188.36 | 187.80 | 189.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 188.36 | 187.80 | 189.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 186.96 | 187.63 | 189.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 186.13 | 187.47 | 189.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 186.35 | 186.72 | 188.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:45:00 | 185.92 | 186.55 | 187.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 176.82 | 182.40 | 185.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 177.03 | 182.40 | 185.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 176.62 | 182.40 | 185.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 179.01 | 177.39 | 180.66 | SL hit (close>ema200) qty=0.50 sl=177.39 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 179.95 | 175.53 | 175.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 186.56 | 180.53 | 178.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 14:15:00 | 194.33 | 194.37 | 192.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 15:00:00 | 194.33 | 194.37 | 192.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 193.45 | 194.11 | 192.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 195.70 | 194.20 | 193.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 15:15:00 | 206.85 | 207.45 | 207.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 206.85 | 207.45 | 207.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 202.36 | 206.43 | 207.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 207.71 | 203.87 | 204.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 207.71 | 203.87 | 204.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 207.71 | 203.87 | 204.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 211.20 | 203.87 | 204.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 210.00 | 205.09 | 205.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 210.00 | 205.09 | 205.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 210.88 | 206.25 | 205.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 211.72 | 207.34 | 206.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 209.53 | 210.90 | 209.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:00:00 | 209.53 | 210.90 | 209.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 209.53 | 210.63 | 209.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:30:00 | 209.26 | 210.63 | 209.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 207.62 | 210.03 | 208.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 207.62 | 210.03 | 208.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 208.00 | 209.62 | 208.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 208.30 | 209.62 | 208.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 207.00 | 209.10 | 208.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 209.85 | 209.10 | 208.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 209.01 | 209.23 | 208.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 209.01 | 209.23 | 208.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 208.89 | 209.16 | 208.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:00:00 | 208.89 | 209.16 | 208.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 208.41 | 209.01 | 208.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:30:00 | 207.40 | 209.01 | 208.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 208.10 | 208.83 | 208.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:45:00 | 208.86 | 208.83 | 208.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 207.40 | 208.54 | 208.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 15:15:00 | 206.50 | 208.13 | 208.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 207.25 | 205.86 | 206.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 15:15:00 | 207.25 | 205.86 | 206.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 207.25 | 205.86 | 206.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 206.82 | 205.86 | 206.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 215.16 | 207.72 | 207.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 10:15:00 | 217.88 | 209.75 | 208.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 15:15:00 | 212.63 | 213.70 | 211.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 202.89 | 213.70 | 211.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 201.41 | 211.24 | 210.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 201.30 | 211.24 | 210.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 200.53 | 209.10 | 209.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 11:15:00 | 199.67 | 207.21 | 208.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 209.27 | 201.57 | 203.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 209.27 | 201.57 | 203.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 209.27 | 201.57 | 203.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 209.27 | 201.57 | 203.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 206.37 | 202.53 | 203.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:15:00 | 204.50 | 202.53 | 203.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:30:00 | 205.19 | 203.42 | 203.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 207.01 | 204.37 | 204.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 207.01 | 204.37 | 204.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 209.09 | 205.86 | 204.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 15:15:00 | 206.80 | 207.11 | 206.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 09:15:00 | 203.75 | 207.11 | 206.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 203.64 | 206.41 | 205.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 203.64 | 206.41 | 205.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 202.62 | 205.65 | 205.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 202.62 | 205.65 | 205.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 202.03 | 204.93 | 205.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 201.83 | 204.31 | 204.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 201.50 | 201.30 | 202.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:00:00 | 201.50 | 201.30 | 202.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 202.62 | 201.59 | 202.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 202.62 | 201.59 | 202.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 202.30 | 201.74 | 202.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 203.60 | 201.74 | 202.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 203.83 | 202.15 | 202.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 204.57 | 202.15 | 202.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 203.76 | 202.48 | 202.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 203.75 | 202.48 | 202.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 207.74 | 203.67 | 203.24 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 202.48 | 204.79 | 204.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 199.19 | 203.67 | 204.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 199.39 | 198.45 | 200.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 199.39 | 198.45 | 200.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 195.23 | 193.75 | 195.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 195.23 | 193.75 | 195.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 196.37 | 194.27 | 195.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 196.37 | 194.27 | 195.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 195.30 | 194.48 | 195.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 191.37 | 194.48 | 195.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 181.80 | 186.74 | 190.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 180.06 | 179.82 | 184.59 | SL hit (close>ema200) qty=0.50 sl=179.82 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 187.80 | 185.53 | 185.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 193.12 | 187.77 | 186.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 15:15:00 | 191.27 | 191.27 | 189.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 09:15:00 | 192.49 | 191.27 | 189.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 191.32 | 194.02 | 192.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 191.32 | 194.02 | 192.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 192.50 | 193.72 | 192.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 192.69 | 193.72 | 192.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 191.33 | 193.24 | 192.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 191.45 | 193.24 | 192.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 190.20 | 192.00 | 192.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 185.90 | 190.78 | 191.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 188.31 | 187.07 | 188.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 188.31 | 187.07 | 188.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 188.31 | 187.07 | 188.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 188.31 | 187.07 | 188.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 190.98 | 187.85 | 188.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 190.98 | 187.85 | 188.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 189.42 | 188.16 | 189.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 188.40 | 189.43 | 189.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 178.98 | 184.60 | 186.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 169.56 | 176.23 | 180.78 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 13:15:00 | 176.25 | 176.17 | 176.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 180.25 | 177.41 | 176.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 179.70 | 182.27 | 180.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 179.70 | 182.27 | 180.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 179.70 | 182.27 | 180.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 179.70 | 182.27 | 180.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 181.80 | 182.17 | 180.60 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 173.43 | 179.52 | 179.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 172.66 | 175.70 | 177.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 178.00 | 175.63 | 177.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 178.00 | 175.63 | 177.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 178.00 | 175.63 | 177.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 178.00 | 175.63 | 177.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 176.30 | 175.77 | 176.99 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 180.00 | 177.40 | 177.26 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 175.50 | 177.88 | 178.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 174.52 | 176.48 | 177.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 163.68 | 163.52 | 167.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 163.68 | 163.52 | 167.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 166.14 | 163.47 | 165.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 166.55 | 163.47 | 165.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 166.49 | 164.07 | 165.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:15:00 | 165.96 | 164.07 | 165.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 165.59 | 164.37 | 165.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 157.66 | 161.19 | 163.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 157.31 | 161.19 | 163.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-18 09:15:00 | 149.36 | 152.49 | 155.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 129 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 158.45 | 155.14 | 155.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 159.35 | 157.30 | 156.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 159.58 | 159.89 | 158.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 159.58 | 159.89 | 158.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 157.18 | 159.47 | 158.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:45:00 | 158.65 | 158.91 | 158.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:00:00 | 158.56 | 158.84 | 158.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 157.86 | 158.65 | 158.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 157.86 | 158.65 | 158.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 156.85 | 158.29 | 158.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 158.54 | 158.07 | 158.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 158.54 | 158.07 | 158.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 158.54 | 158.07 | 158.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 158.91 | 158.07 | 158.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 157.45 | 157.94 | 158.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 156.70 | 157.94 | 158.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 148.86 | 151.39 | 153.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 12:15:00 | 141.03 | 147.41 | 151.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 131 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 148.30 | 144.99 | 144.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 148.65 | 145.72 | 145.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 149.10 | 149.68 | 148.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 151.90 | 149.68 | 148.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 152.53 | 150.25 | 148.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 152.88 | 150.25 | 148.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 13:15:00 | 147.67 | 149.45 | 149.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 147.67 | 149.45 | 149.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 147.20 | 149.00 | 149.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 147.90 | 147.00 | 147.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 147.90 | 147.00 | 147.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 147.90 | 147.00 | 147.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 147.30 | 147.00 | 147.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 146.31 | 146.86 | 147.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 145.91 | 146.86 | 147.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 145.20 | 146.72 | 147.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 144.98 | 145.25 | 146.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 145.40 | 144.60 | 144.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 145.40 | 144.60 | 144.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 146.85 | 145.05 | 144.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 162.16 | 164.11 | 161.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 162.16 | 164.11 | 161.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 161.25 | 163.54 | 161.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 161.76 | 163.54 | 161.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 161.79 | 163.19 | 161.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 160.33 | 163.19 | 161.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 161.69 | 162.89 | 161.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 162.43 | 162.75 | 161.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 160.10 | 161.99 | 161.25 | SL hit (close<static) qty=1.00 sl=160.80 alert=retest2 |

### Cycle 134 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 160.01 | 160.87 | 160.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 158.53 | 160.40 | 160.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 159.85 | 159.82 | 160.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 159.85 | 159.82 | 160.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 159.85 | 159.82 | 160.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 159.85 | 159.82 | 160.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 160.11 | 159.88 | 160.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 160.95 | 159.88 | 160.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 159.57 | 159.82 | 160.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:30:00 | 161.26 | 159.82 | 160.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 160.35 | 159.08 | 159.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:30:00 | 158.35 | 158.91 | 159.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 157.08 | 158.91 | 159.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 158.20 | 157.40 | 157.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:00:00 | 158.22 | 157.56 | 157.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 159.30 | 158.10 | 157.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 159.30 | 158.10 | 157.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 160.42 | 158.57 | 158.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 157.16 | 160.80 | 159.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 157.16 | 160.80 | 159.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 157.16 | 160.80 | 159.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 157.16 | 160.80 | 159.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 157.33 | 160.11 | 159.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 11:15:00 | 156.52 | 160.11 | 159.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 155.39 | 159.17 | 159.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 145.86 | 154.80 | 156.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 151.65 | 150.57 | 153.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 153.41 | 150.57 | 153.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 151.54 | 150.77 | 153.22 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 157.23 | 154.09 | 153.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 160.34 | 156.76 | 155.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 171.10 | 171.19 | 167.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 09:15:00 | 169.40 | 171.19 | 167.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 172.26 | 172.66 | 171.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 174.30 | 172.66 | 171.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 171.10 | 172.35 | 171.37 | SL hit (close<static) qty=1.00 sl=171.11 alert=retest2 |

### Cycle 138 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 169.73 | 172.32 | 172.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 167.10 | 170.31 | 171.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 12:15:00 | 170.76 | 170.13 | 171.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 12:15:00 | 170.76 | 170.13 | 171.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 170.76 | 170.13 | 171.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:00:00 | 170.76 | 170.13 | 171.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 170.50 | 170.20 | 170.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:45:00 | 170.52 | 170.20 | 170.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 170.81 | 170.33 | 170.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 170.81 | 170.33 | 170.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 170.55 | 170.37 | 170.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 172.60 | 170.37 | 170.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 171.27 | 170.55 | 170.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:00:00 | 170.20 | 170.51 | 170.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 175.11 | 171.78 | 171.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 175.11 | 171.78 | 171.38 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 170.10 | 171.82 | 171.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 10:15:00 | 169.84 | 171.16 | 171.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 15:15:00 | 170.75 | 170.74 | 171.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:15:00 | 170.42 | 170.74 | 171.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 167.90 | 170.17 | 170.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:15:00 | 167.14 | 169.17 | 170.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:00:00 | 166.13 | 168.56 | 169.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 166.64 | 167.55 | 167.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 158.78 | 165.07 | 166.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 10:15:00 | 158.31 | 164.00 | 165.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 167.37 | 162.59 | 164.07 | SL hit (close>ema200) qty=0.50 sl=162.59 alert=retest2 |

### Cycle 141 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 168.58 | 165.48 | 165.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 170.84 | 167.85 | 166.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 171.75 | 172.12 | 170.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:30:00 | 171.44 | 172.12 | 170.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 178.49 | 180.36 | 178.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 178.49 | 180.36 | 178.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 182.81 | 180.85 | 178.95 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 174.77 | 178.04 | 178.23 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 179.00 | 178.27 | 178.21 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 177.75 | 178.16 | 178.17 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 179.00 | 178.33 | 178.25 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 177.00 | 178.03 | 178.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 176.70 | 177.76 | 177.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 180.61 | 177.92 | 177.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 180.61 | 177.92 | 177.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 180.61 | 177.92 | 177.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 181.04 | 177.92 | 177.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 186.69 | 179.67 | 178.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 188.20 | 183.69 | 182.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 185.78 | 185.87 | 184.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:30:00 | 185.68 | 185.87 | 184.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 184.00 | 185.94 | 185.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 183.85 | 185.94 | 185.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 183.72 | 185.50 | 185.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 183.03 | 185.50 | 185.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 187.16 | 185.84 | 185.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 188.15 | 186.76 | 186.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 184.72 | 185.97 | 186.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 184.72 | 185.97 | 186.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 184.00 | 185.42 | 185.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 186.21 | 184.96 | 185.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 186.21 | 184.96 | 185.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 186.21 | 184.96 | 185.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 186.50 | 184.96 | 185.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 185.30 | 185.03 | 185.35 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 190.50 | 186.33 | 185.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 193.99 | 192.12 | 190.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 197.72 | 197.90 | 195.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:45:00 | 197.85 | 197.90 | 195.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 194.06 | 196.92 | 195.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 194.06 | 196.92 | 195.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 194.38 | 196.41 | 195.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 194.38 | 196.41 | 195.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 191.22 | 194.51 | 194.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 188.50 | 192.86 | 193.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 187.07 | 186.98 | 188.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:15:00 | 186.94 | 186.98 | 188.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 187.79 | 187.13 | 188.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 186.03 | 186.57 | 188.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 176.73 | 180.26 | 182.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 180.48 | 179.39 | 181.47 | SL hit (close>ema200) qty=0.50 sl=179.39 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 185.92 | 182.19 | 182.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 186.13 | 182.98 | 182.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 186.79 | 187.43 | 185.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 186.79 | 187.43 | 185.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 187.10 | 188.65 | 187.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 187.10 | 188.65 | 187.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 186.87 | 188.29 | 187.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 186.87 | 188.29 | 187.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 185.69 | 187.77 | 187.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 185.83 | 187.77 | 187.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 185.26 | 187.27 | 187.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 10:15:00 | 183.71 | 185.84 | 186.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 186.18 | 184.70 | 185.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 186.18 | 184.70 | 185.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 186.18 | 184.70 | 185.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 186.18 | 184.70 | 185.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 189.17 | 185.60 | 185.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 191.44 | 185.60 | 185.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 190.08 | 186.49 | 186.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 192.47 | 188.59 | 187.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 188.94 | 189.41 | 188.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:00:00 | 188.94 | 189.41 | 188.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 187.55 | 189.82 | 188.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 187.55 | 189.82 | 188.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 188.32 | 189.52 | 188.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 188.78 | 189.52 | 188.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 189.15 | 189.18 | 188.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 187.73 | 188.64 | 188.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 187.73 | 188.64 | 188.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 187.52 | 188.41 | 188.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 186.34 | 185.80 | 186.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:45:00 | 186.34 | 185.80 | 186.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 184.88 | 185.61 | 186.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 184.65 | 185.22 | 186.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 188.00 | 185.81 | 186.12 | SL hit (close>static) qty=1.00 sl=187.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 188.79 | 186.65 | 186.42 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 184.98 | 186.80 | 187.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 183.59 | 186.15 | 186.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 184.10 | 183.68 | 184.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 184.10 | 183.68 | 184.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 183.37 | 183.62 | 184.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:30:00 | 184.10 | 183.62 | 184.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 184.69 | 183.75 | 184.46 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 184.90 | 184.73 | 184.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 186.99 | 185.18 | 184.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 185.70 | 186.08 | 185.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 185.70 | 186.08 | 185.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 185.70 | 186.08 | 185.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 186.53 | 185.73 | 185.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 184.05 | 185.33 | 185.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 184.05 | 185.33 | 185.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 183.70 | 184.29 | 184.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 183.97 | 183.81 | 184.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 183.97 | 183.81 | 184.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 183.97 | 183.81 | 184.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 183.97 | 183.81 | 184.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 183.50 | 183.75 | 184.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 183.00 | 183.75 | 184.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 184.45 | 183.62 | 184.08 | SL hit (close>static) qty=1.00 sl=184.25 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 185.35 | 184.20 | 184.17 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 183.95 | 184.21 | 184.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 183.47 | 184.07 | 184.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 173.93 | 173.67 | 176.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:30:00 | 173.10 | 173.67 | 176.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 186.80 | 176.24 | 177.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 186.00 | 176.24 | 177.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 189.79 | 178.95 | 178.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 193.30 | 183.43 | 180.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 196.45 | 201.18 | 195.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 15:15:00 | 196.45 | 201.18 | 195.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 196.45 | 201.18 | 195.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 195.25 | 200.20 | 195.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 195.63 | 199.28 | 195.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 195.45 | 199.28 | 195.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 191.80 | 197.79 | 194.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 191.80 | 197.79 | 194.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 193.00 | 196.83 | 194.69 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 188.49 | 193.20 | 193.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 184.97 | 191.55 | 192.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 200.78 | 190.53 | 190.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 200.78 | 190.53 | 190.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 200.78 | 190.53 | 190.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 202.69 | 190.53 | 190.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 201.82 | 192.79 | 191.97 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 191.58 | 193.83 | 193.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 189.67 | 192.58 | 193.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 193.00 | 190.15 | 191.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 193.00 | 190.15 | 191.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 193.00 | 190.15 | 191.40 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 193.50 | 192.20 | 192.09 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 189.90 | 191.64 | 191.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 187.51 | 190.49 | 191.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 15:15:00 | 187.00 | 185.09 | 186.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 15:15:00 | 187.00 | 185.09 | 186.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 187.00 | 185.09 | 186.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 190.88 | 185.09 | 186.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 197.17 | 187.50 | 187.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 204.53 | 195.32 | 193.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 198.40 | 199.23 | 196.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 195.40 | 199.23 | 196.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 196.73 | 198.73 | 196.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 195.60 | 198.73 | 196.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 195.80 | 198.14 | 196.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 195.81 | 198.14 | 196.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 196.22 | 197.76 | 196.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 196.50 | 197.76 | 196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 196.00 | 197.41 | 196.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 196.15 | 197.41 | 196.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 196.30 | 196.81 | 196.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 195.00 | 196.81 | 196.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 193.91 | 196.23 | 196.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 193.73 | 195.07 | 195.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 191.10 | 190.91 | 192.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:45:00 | 191.11 | 190.91 | 192.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 191.81 | 191.06 | 191.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 192.50 | 191.06 | 191.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 192.82 | 191.41 | 192.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 189.25 | 191.41 | 192.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 190.86 | 189.66 | 190.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 193.30 | 190.63 | 190.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 193.30 | 190.63 | 190.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 195.54 | 192.70 | 191.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 199.10 | 199.35 | 196.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:00:00 | 199.10 | 199.35 | 196.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 196.00 | 198.71 | 197.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 196.00 | 198.71 | 197.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 196.17 | 198.21 | 197.00 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 194.95 | 196.22 | 196.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 193.32 | 195.27 | 195.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 193.46 | 192.04 | 193.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 193.46 | 192.04 | 193.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 193.46 | 192.04 | 193.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 193.06 | 192.24 | 193.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 192.85 | 192.66 | 192.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 193.60 | 192.71 | 192.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 193.60 | 192.71 | 192.65 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 192.20 | 192.78 | 192.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 192.08 | 192.64 | 192.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 12:15:00 | 192.80 | 192.68 | 192.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 12:15:00 | 192.80 | 192.68 | 192.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 192.80 | 192.68 | 192.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 192.80 | 192.68 | 192.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 192.74 | 192.69 | 192.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:30:00 | 193.21 | 192.69 | 192.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 192.61 | 192.67 | 192.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 192.86 | 192.67 | 192.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 192.46 | 192.63 | 192.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 192.88 | 192.63 | 192.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 192.03 | 192.51 | 192.68 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 194.56 | 192.92 | 192.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 201.05 | 196.27 | 194.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 198.40 | 198.76 | 197.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 15:00:00 | 198.40 | 198.76 | 197.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 198.15 | 198.64 | 197.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 196.67 | 198.45 | 197.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 196.40 | 198.04 | 197.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 196.80 | 198.04 | 197.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 197.05 | 197.84 | 197.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 196.45 | 197.84 | 197.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 197.30 | 197.73 | 197.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:45:00 | 196.92 | 197.73 | 197.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 196.75 | 197.54 | 197.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 196.75 | 197.54 | 197.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 195.69 | 197.17 | 197.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 196.22 | 197.17 | 197.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 196.00 | 196.93 | 196.95 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 197.80 | 197.11 | 197.02 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 195.54 | 196.79 | 196.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 195.00 | 196.43 | 196.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 196.73 | 196.08 | 196.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 196.73 | 196.08 | 196.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 196.73 | 196.08 | 196.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 195.20 | 195.98 | 196.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 195.61 | 195.80 | 196.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 185.83 | 188.42 | 190.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 188.78 | 188.22 | 189.99 | SL hit (close>ema200) qty=0.50 sl=188.22 alert=retest2 |

### Cycle 177 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 191.04 | 189.54 | 189.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 192.11 | 190.40 | 190.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 191.86 | 191.97 | 191.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 191.86 | 191.97 | 191.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 191.13 | 191.81 | 191.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 191.05 | 191.81 | 191.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 190.70 | 191.58 | 191.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 190.93 | 191.58 | 191.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 191.43 | 191.55 | 191.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 193.10 | 191.76 | 191.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 190.33 | 191.43 | 191.20 | SL hit (close<static) qty=1.00 sl=190.70 alert=retest2 |

### Cycle 178 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 189.54 | 190.76 | 190.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 189.10 | 190.20 | 190.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 189.45 | 189.43 | 189.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 190.32 | 189.43 | 189.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 191.00 | 189.75 | 190.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 190.71 | 189.75 | 190.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 189.87 | 189.77 | 189.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 190.14 | 189.77 | 189.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 189.37 | 189.69 | 189.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:30:00 | 189.99 | 189.69 | 189.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 192.71 | 190.13 | 190.06 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 188.78 | 190.14 | 190.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 187.75 | 189.66 | 190.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 188.76 | 188.70 | 189.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 188.76 | 188.70 | 189.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 188.76 | 188.70 | 189.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 188.00 | 188.54 | 188.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 188.78 | 187.59 | 187.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 188.78 | 187.59 | 187.53 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 187.25 | 187.61 | 187.65 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 190.11 | 187.85 | 187.70 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 189.05 | 189.66 | 189.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 188.96 | 189.52 | 189.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 184.80 | 183.97 | 185.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 184.80 | 183.97 | 185.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 184.80 | 183.97 | 185.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 183.48 | 184.19 | 185.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:45:00 | 183.38 | 184.06 | 185.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 182.20 | 184.06 | 185.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 183.43 | 183.41 | 184.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 182.21 | 181.54 | 182.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 182.21 | 181.54 | 182.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 182.87 | 181.80 | 182.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 183.00 | 181.80 | 182.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 182.24 | 181.89 | 182.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 183.21 | 182.86 | 182.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 183.21 | 182.86 | 182.81 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 181.05 | 182.58 | 182.70 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 184.90 | 182.81 | 182.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 185.60 | 184.33 | 183.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 184.39 | 184.70 | 184.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 184.39 | 184.70 | 184.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 184.39 | 184.70 | 184.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 183.73 | 184.70 | 184.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 184.15 | 184.59 | 184.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 184.15 | 184.59 | 184.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 184.69 | 184.61 | 184.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 179.75 | 184.61 | 184.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 179.86 | 183.66 | 183.79 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 182.29 | 182.02 | 182.00 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 181.75 | 181.98 | 181.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 179.62 | 181.51 | 181.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 176.51 | 175.74 | 177.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 176.51 | 175.74 | 177.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 176.51 | 175.74 | 177.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 175.03 | 175.80 | 176.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 166.28 | 169.54 | 170.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 09:15:00 | 157.53 | 160.11 | 162.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 161.30 | 159.63 | 159.62 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 159.51 | 159.82 | 159.86 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 164.06 | 160.57 | 160.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 171.00 | 164.31 | 162.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 166.99 | 167.04 | 165.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:15:00 | 169.71 | 167.04 | 165.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 166.37 | 167.30 | 166.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 166.37 | 167.30 | 166.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 166.90 | 167.22 | 166.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 165.80 | 167.22 | 166.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 164.31 | 166.64 | 166.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 164.31 | 166.64 | 166.13 | SL hit (close<ema400) qty=1.00 sl=166.13 alert=retest1 |

### Cycle 194 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 163.21 | 165.47 | 165.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 163.00 | 164.97 | 165.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 155.11 | 154.64 | 156.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 155.22 | 154.64 | 156.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 156.29 | 153.22 | 154.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 156.89 | 153.22 | 154.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 154.29 | 153.43 | 154.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 156.87 | 153.43 | 154.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 154.03 | 153.55 | 154.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 153.85 | 153.60 | 154.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 153.90 | 153.72 | 153.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 156.67 | 153.34 | 153.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 156.67 | 153.34 | 153.28 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 154.10 | 155.32 | 155.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 153.64 | 154.75 | 155.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 154.05 | 153.41 | 153.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 14:15:00 | 154.05 | 153.41 | 153.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 154.05 | 153.41 | 153.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 154.40 | 153.41 | 153.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 154.00 | 153.53 | 153.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 153.29 | 153.53 | 153.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 153.13 | 153.45 | 153.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 151.84 | 153.06 | 153.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 151.08 | 150.39 | 150.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 151.08 | 150.39 | 150.37 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 14:15:00 | 150.09 | 150.31 | 150.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 149.19 | 149.86 | 150.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 149.92 | 149.88 | 150.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 149.92 | 149.88 | 150.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 150.30 | 149.96 | 150.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 150.30 | 149.96 | 150.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 150.15 | 150.00 | 150.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 149.82 | 150.00 | 150.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 149.60 | 150.00 | 150.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 150.86 | 149.99 | 150.05 | SL hit (close>static) qty=1.00 sl=150.60 alert=retest2 |

### Cycle 199 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 150.53 | 150.10 | 150.09 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 149.98 | 150.07 | 150.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 148.37 | 149.71 | 149.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 145.98 | 143.84 | 144.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 145.98 | 143.84 | 144.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 145.98 | 143.84 | 144.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 146.21 | 143.84 | 144.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 145.19 | 144.11 | 144.93 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 146.58 | 145.35 | 145.33 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 144.31 | 145.26 | 145.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 143.25 | 144.86 | 145.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 144.85 | 143.73 | 144.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 144.85 | 143.73 | 144.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 144.85 | 143.73 | 144.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 145.52 | 143.73 | 144.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 144.62 | 143.91 | 144.22 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 146.80 | 144.83 | 144.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 147.45 | 145.87 | 145.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 146.40 | 146.44 | 145.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 11:00:00 | 146.40 | 146.44 | 145.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 147.11 | 146.75 | 146.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 145.39 | 146.75 | 146.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 147.80 | 146.96 | 146.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 149.77 | 147.76 | 146.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 142.94 | 147.33 | 147.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 142.94 | 147.33 | 147.89 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 147.65 | 147.18 | 147.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 148.79 | 147.62 | 147.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 147.80 | 148.39 | 147.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 147.80 | 148.39 | 147.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 147.80 | 148.39 | 147.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 148.17 | 148.39 | 147.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 146.80 | 148.07 | 147.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 146.80 | 148.07 | 147.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 146.72 | 147.80 | 147.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 147.15 | 147.73 | 147.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 147.18 | 147.62 | 147.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 147.18 | 147.62 | 147.65 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 15:15:00 | 148.00 | 147.71 | 147.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 10:15:00 | 150.95 | 148.56 | 148.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 156.84 | 157.21 | 154.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 155.12 | 156.34 | 155.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 155.12 | 156.34 | 155.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 153.91 | 156.34 | 155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 154.83 | 156.04 | 155.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 154.83 | 156.04 | 155.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 153.70 | 155.57 | 155.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 153.70 | 155.57 | 155.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 154.63 | 154.97 | 154.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 151.06 | 154.19 | 154.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 152.45 | 151.21 | 151.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 152.45 | 151.21 | 151.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 152.45 | 151.21 | 151.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 152.15 | 151.21 | 151.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 152.15 | 151.40 | 151.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 152.89 | 151.40 | 151.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 152.00 | 151.96 | 152.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 152.07 | 151.96 | 152.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 151.75 | 151.91 | 152.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 151.17 | 151.91 | 152.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:30:00 | 151.50 | 151.63 | 151.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:00:00 | 151.54 | 151.63 | 151.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 154.50 | 152.33 | 152.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 154.50 | 152.33 | 152.13 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 150.74 | 151.90 | 151.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 150.61 | 151.42 | 151.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 152.45 | 150.81 | 151.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 152.45 | 150.81 | 151.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 152.45 | 150.81 | 151.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 152.06 | 150.81 | 151.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 151.70 | 150.99 | 151.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 151.12 | 151.13 | 151.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 152.08 | 151.32 | 151.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 152.08 | 151.32 | 151.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 152.24 | 151.50 | 151.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 151.13 | 151.43 | 151.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 151.13 | 151.43 | 151.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 151.13 | 151.43 | 151.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 150.70 | 151.43 | 151.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 151.25 | 151.39 | 151.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:45:00 | 150.76 | 151.39 | 151.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 150.50 | 151.21 | 151.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 149.36 | 150.84 | 151.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 150.99 | 150.35 | 150.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 150.99 | 150.35 | 150.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 150.99 | 150.35 | 150.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 150.99 | 150.35 | 150.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 149.80 | 150.24 | 150.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 149.67 | 150.20 | 150.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 149.70 | 149.53 | 149.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 149.65 | 149.37 | 149.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 142.19 | 146.56 | 147.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 142.21 | 146.56 | 147.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 142.17 | 146.56 | 147.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 139.75 | 138.81 | 140.49 | SL hit (close>ema200) qty=0.50 sl=138.81 alert=retest2 |

### Cycle 213 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 138.29 | 137.42 | 137.36 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 136.74 | 137.42 | 137.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 136.55 | 137.25 | 137.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 137.10 | 136.82 | 137.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 137.10 | 136.82 | 137.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 137.10 | 136.82 | 137.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 137.10 | 136.82 | 137.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 138.20 | 137.09 | 137.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 138.20 | 137.09 | 137.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 138.48 | 137.37 | 137.32 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 136.77 | 137.31 | 137.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 135.26 | 136.81 | 137.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 133.07 | 132.88 | 134.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:30:00 | 132.85 | 132.88 | 134.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 134.29 | 133.38 | 133.85 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 135.45 | 134.12 | 134.10 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 132.05 | 133.91 | 134.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 130.50 | 132.48 | 133.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 132.29 | 131.77 | 132.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 132.29 | 131.77 | 132.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 132.29 | 131.77 | 132.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 132.68 | 131.77 | 132.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 132.63 | 131.94 | 132.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 132.56 | 131.94 | 132.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 132.41 | 132.04 | 132.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 131.90 | 132.04 | 132.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 131.90 | 132.01 | 132.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 131.71 | 131.93 | 132.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 128.44 | 132.29 | 132.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 125.30 | 129.79 | 131.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 125.30 | 129.79 | 131.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 125.12 | 129.79 | 131.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:15:00 | 122.02 | 125.26 | 127.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 124.78 | 124.76 | 126.97 | SL hit (close>ema200) qty=0.50 sl=124.76 alert=retest2 |

### Cycle 219 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 128.03 | 127.41 | 127.36 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 124.51 | 126.93 | 127.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 124.00 | 126.34 | 126.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 123.03 | 120.07 | 122.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 123.03 | 120.07 | 122.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 123.03 | 120.07 | 122.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 123.54 | 120.07 | 122.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 122.15 | 120.49 | 122.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 119.11 | 122.64 | 122.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 123.38 | 121.90 | 122.05 | SL hit (close>static) qty=1.00 sl=123.14 alert=retest2 |

### Cycle 221 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 123.50 | 122.22 | 122.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 123.68 | 122.74 | 122.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 123.19 | 123.46 | 122.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 123.19 | 123.46 | 122.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 123.19 | 123.46 | 122.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 123.28 | 123.46 | 122.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 165.86 | 167.79 | 166.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 167.30 | 167.61 | 166.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 164.00 | 165.82 | 165.77 | SL hit (close<static) qty=1.00 sl=164.13 alert=retest2 |

### Cycle 222 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 165.30 | 165.71 | 165.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 163.96 | 165.01 | 165.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 166.89 | 164.23 | 164.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 166.89 | 164.23 | 164.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 166.89 | 164.23 | 164.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 166.89 | 164.23 | 164.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 170.48 | 165.48 | 165.04 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 163.13 | 165.17 | 165.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 161.89 | 163.57 | 164.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 163.93 | 163.64 | 164.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 163.93 | 163.64 | 164.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 164.10 | 163.73 | 164.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 163.45 | 163.73 | 164.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 164.52 | 163.89 | 164.32 | EMA400 retest candle locked (from downside) |

### Cycle 225 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 165.36 | 164.65 | 164.59 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 164.00 | 164.62 | 164.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 163.70 | 164.44 | 164.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 164.12 | 164.05 | 164.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 164.12 | 164.05 | 164.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 164.12 | 164.05 | 164.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 162.39 | 163.61 | 164.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 162.30 | 161.28 | 161.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 162.17 | 161.53 | 161.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 162.17 | 161.53 | 161.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 162.64 | 161.76 | 161.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 161.22 | 161.72 | 161.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 161.22 | 161.72 | 161.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 161.22 | 161.72 | 161.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 161.22 | 161.72 | 161.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 162.20 | 161.82 | 161.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 162.95 | 162.01 | 161.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 163.17 | 162.29 | 161.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 163.26 | 162.29 | 161.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:15:00 | 228.45 | 2024-04-15 09:15:00 | 217.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 09:15:00 | 228.45 | 2024-04-16 09:15:00 | 221.50 | STOP_HIT | 0.50 | 3.04% |
| BUY | retest2 | 2024-04-25 10:15:00 | 227.40 | 2024-04-29 09:15:00 | 250.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-25 11:15:00 | 227.45 | 2024-04-29 09:15:00 | 250.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-25 15:15:00 | 227.30 | 2024-04-29 09:15:00 | 250.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-26 11:15:00 | 228.50 | 2024-04-29 09:15:00 | 251.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-22 09:15:00 | 240.45 | 2024-05-24 14:15:00 | 239.50 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-05-30 09:45:00 | 230.35 | 2024-06-03 10:15:00 | 234.65 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-05-31 09:30:00 | 229.25 | 2024-06-03 10:15:00 | 234.65 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-06-27 12:30:00 | 240.19 | 2024-07-01 10:15:00 | 246.25 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-07-02 15:15:00 | 242.72 | 2024-07-05 09:15:00 | 266.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 10:15:00 | 242.07 | 2024-07-05 09:15:00 | 266.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 11:00:00 | 241.88 | 2024-07-05 09:15:00 | 266.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 13:30:00 | 242.10 | 2024-07-05 09:15:00 | 266.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-10 13:45:00 | 276.93 | 2024-07-18 12:15:00 | 281.05 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2024-07-25 13:30:00 | 258.82 | 2024-07-26 11:15:00 | 279.89 | STOP_HIT | 1.00 | -8.14% |
| SELL | retest2 | 2024-07-26 09:30:00 | 258.43 | 2024-07-26 11:15:00 | 279.89 | STOP_HIT | 1.00 | -8.30% |
| SELL | retest2 | 2024-07-26 10:15:00 | 258.87 | 2024-07-26 11:15:00 | 279.89 | STOP_HIT | 1.00 | -8.12% |
| SELL | retest2 | 2024-07-26 11:00:00 | 258.99 | 2024-07-26 11:15:00 | 279.89 | STOP_HIT | 1.00 | -8.07% |
| BUY | retest2 | 2024-07-30 09:15:00 | 282.61 | 2024-08-01 15:15:00 | 280.25 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-07-30 10:45:00 | 282.28 | 2024-08-01 15:15:00 | 280.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2024-08-06 13:30:00 | 261.55 | 2024-08-07 15:15:00 | 264.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-08 15:15:00 | 259.90 | 2024-08-12 09:15:00 | 246.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 15:15:00 | 259.90 | 2024-08-14 09:15:00 | 233.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-27 09:15:00 | 268.00 | 2024-08-29 11:15:00 | 262.35 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-09-02 09:30:00 | 262.60 | 2024-09-02 12:15:00 | 268.35 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-09-02 10:00:00 | 262.25 | 2024-09-02 12:15:00 | 268.35 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-09-02 11:00:00 | 260.20 | 2024-09-02 12:15:00 | 268.35 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-09-06 10:30:00 | 255.65 | 2024-09-11 13:15:00 | 242.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 11:00:00 | 255.65 | 2024-09-11 13:15:00 | 242.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 14:15:00 | 254.80 | 2024-09-11 13:15:00 | 242.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-10 09:15:00 | 253.40 | 2024-09-11 13:15:00 | 240.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 10:30:00 | 255.65 | 2024-09-12 09:15:00 | 244.40 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2024-09-06 11:00:00 | 255.65 | 2024-09-12 09:15:00 | 244.40 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2024-09-06 14:15:00 | 254.80 | 2024-09-12 09:15:00 | 244.40 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2024-09-10 09:15:00 | 253.40 | 2024-09-12 09:15:00 | 244.40 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2024-09-10 11:00:00 | 248.25 | 2024-09-18 13:15:00 | 235.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-10 11:00:00 | 248.25 | 2024-09-19 14:15:00 | 234.70 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2024-10-03 09:15:00 | 230.10 | 2024-10-07 09:15:00 | 218.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:45:00 | 230.11 | 2024-10-07 09:15:00 | 218.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 230.10 | 2024-10-07 13:15:00 | 207.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-03 09:45:00 | 230.11 | 2024-10-07 13:15:00 | 207.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 11:15:00 | 209.30 | 2024-10-22 12:15:00 | 198.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:15:00 | 209.30 | 2024-10-23 10:15:00 | 199.94 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2024-11-11 13:15:00 | 186.13 | 2024-11-13 09:15:00 | 176.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:30:00 | 186.35 | 2024-11-13 09:15:00 | 177.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:45:00 | 185.92 | 2024-11-13 09:15:00 | 176.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:15:00 | 186.13 | 2024-11-14 09:15:00 | 179.01 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2024-11-12 09:30:00 | 186.35 | 2024-11-14 09:15:00 | 179.01 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2024-11-12 10:45:00 | 185.92 | 2024-11-14 09:15:00 | 179.01 | STOP_HIT | 0.50 | 3.72% |
| BUY | retest2 | 2024-12-02 15:15:00 | 195.70 | 2024-12-12 15:15:00 | 206.85 | STOP_HIT | 1.00 | 5.70% |
| SELL | retest2 | 2024-12-26 11:15:00 | 204.50 | 2024-12-26 14:15:00 | 207.01 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-12-26 12:30:00 | 205.19 | 2024-12-26 14:15:00 | 207.01 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-01-10 09:15:00 | 191.37 | 2025-01-13 11:15:00 | 181.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 191.37 | 2025-01-14 10:15:00 | 180.06 | STOP_HIT | 0.50 | 5.91% |
| SELL | retest2 | 2025-01-24 09:15:00 | 188.40 | 2025-01-27 09:15:00 | 178.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 188.40 | 2025-01-28 09:15:00 | 169.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 165.96 | 2025-02-14 10:15:00 | 157.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:00:00 | 165.59 | 2025-02-14 10:15:00 | 157.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 165.96 | 2025-02-18 09:15:00 | 149.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 12:00:00 | 165.59 | 2025-02-18 09:15:00 | 149.03 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-24 11:45:00 | 158.65 | 2025-02-24 13:15:00 | 157.86 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-02-24 13:00:00 | 158.56 | 2025-02-24 13:15:00 | 157.86 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-02-25 11:15:00 | 156.70 | 2025-02-28 09:15:00 | 148.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:15:00 | 156.70 | 2025-02-28 12:15:00 | 141.03 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-07 10:15:00 | 152.88 | 2025-03-10 13:15:00 | 147.67 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-03-12 10:15:00 | 145.91 | 2025-03-18 13:15:00 | 145.40 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-03-12 11:15:00 | 145.20 | 2025-03-18 13:15:00 | 145.40 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-03-13 10:45:00 | 144.98 | 2025-03-18 13:15:00 | 145.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-03-25 14:15:00 | 162.43 | 2025-03-25 15:15:00 | 160.10 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-03-26 09:15:00 | 164.58 | 2025-03-26 10:15:00 | 160.40 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-03-28 12:30:00 | 158.35 | 2025-04-02 15:15:00 | 159.30 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-03-28 13:00:00 | 157.08 | 2025-04-02 15:15:00 | 159.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-04-02 12:45:00 | 158.20 | 2025-04-02 15:15:00 | 159.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-04-02 14:00:00 | 158.22 | 2025-04-02 15:15:00 | 159.30 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-04-23 09:15:00 | 174.30 | 2025-04-23 09:15:00 | 171.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-04-23 12:45:00 | 173.10 | 2025-04-25 10:15:00 | 168.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-04-25 10:00:00 | 172.96 | 2025-04-25 10:15:00 | 168.80 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-04-29 12:00:00 | 170.20 | 2025-04-30 09:15:00 | 175.11 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-06 13:15:00 | 167.14 | 2025-05-09 09:15:00 | 158.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 14:00:00 | 166.13 | 2025-05-09 10:15:00 | 158.31 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-05-06 13:15:00 | 167.14 | 2025-05-12 09:15:00 | 167.37 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2025-05-06 14:00:00 | 166.13 | 2025-05-12 09:15:00 | 167.37 | STOP_HIT | 0.50 | -0.75% |
| SELL | retest2 | 2025-05-08 13:30:00 | 166.64 | 2025-05-12 12:15:00 | 168.58 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-05-12 09:45:00 | 167.41 | 2025-05-12 12:15:00 | 168.58 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-06-02 15:15:00 | 188.15 | 2025-06-03 13:15:00 | 184.72 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-17 11:30:00 | 186.03 | 2025-06-19 13:15:00 | 176.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 186.03 | 2025-06-20 09:15:00 | 180.48 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2025-07-02 11:15:00 | 188.78 | 2025-07-03 10:15:00 | 187.73 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-02 14:45:00 | 189.15 | 2025-07-03 10:15:00 | 187.73 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-07 11:30:00 | 184.65 | 2025-07-07 15:15:00 | 188.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-17 14:30:00 | 186.53 | 2025-07-18 09:15:00 | 184.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-22 13:15:00 | 183.00 | 2025-07-22 14:15:00 | 184.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-29 09:15:00 | 189.25 | 2025-09-02 09:15:00 | 193.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-09-01 15:00:00 | 190.86 | 2025-09-02 09:15:00 | 193.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-10 11:00:00 | 193.06 | 2025-09-15 09:15:00 | 193.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-09-11 10:45:00 | 192.85 | 2025-09-15 09:15:00 | 193.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-25 10:30:00 | 195.20 | 2025-09-29 12:15:00 | 185.83 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-09-25 10:30:00 | 195.20 | 2025-09-30 09:15:00 | 188.78 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-09-25 12:45:00 | 195.61 | 2025-10-03 09:15:00 | 191.04 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-10-07 14:45:00 | 193.10 | 2025-10-08 09:15:00 | 190.33 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-17 11:15:00 | 188.00 | 2025-10-23 10:15:00 | 188.78 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-04 14:15:00 | 183.48 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-11-04 14:45:00 | 183.38 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-11-04 15:15:00 | 182.20 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-06 09:30:00 | 183.43 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-11-24 14:15:00 | 175.03 | 2025-12-02 09:15:00 | 166.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 14:15:00 | 175.03 | 2025-12-08 09:15:00 | 157.53 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-12-16 09:15:00 | 169.71 | 2025-12-17 09:15:00 | 164.31 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-12-26 13:30:00 | 153.85 | 2025-12-31 09:15:00 | 156.67 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-29 10:30:00 | 153.90 | 2025-12-31 09:15:00 | 156.67 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-08 10:45:00 | 151.84 | 2026-01-13 12:15:00 | 151.08 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2026-01-14 14:15:00 | 149.82 | 2026-01-16 11:15:00 | 150.86 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-16 09:15:00 | 149.60 | 2026-01-16 11:15:00 | 150.86 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-30 10:30:00 | 149.77 | 2026-02-02 10:15:00 | 142.94 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-02-05 12:30:00 | 147.15 | 2026-02-05 13:15:00 | 147.18 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2026-02-18 10:15:00 | 151.17 | 2026-02-18 15:15:00 | 154.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-02-18 13:30:00 | 151.50 | 2026-02-18 15:15:00 | 154.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-02-18 14:00:00 | 151.54 | 2026-02-18 15:15:00 | 154.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-02-23 13:30:00 | 151.12 | 2026-02-23 14:15:00 | 152.08 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-25 12:15:00 | 149.67 | 2026-03-02 09:15:00 | 142.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 149.70 | 2026-03-02 09:15:00 | 142.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 149.65 | 2026-03-02 09:15:00 | 142.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:15:00 | 149.67 | 2026-03-05 14:15:00 | 139.75 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-02-26 10:45:00 | 149.70 | 2026-03-05 14:15:00 | 139.75 | STOP_HIT | 0.50 | 6.65% |
| SELL | retest2 | 2026-02-26 11:30:00 | 149.65 | 2026-03-05 14:15:00 | 139.75 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2026-03-20 12:15:00 | 131.90 | 2026-03-23 10:15:00 | 125.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:00:00 | 131.90 | 2026-03-23 10:15:00 | 125.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 131.71 | 2026-03-23 10:15:00 | 125.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 128.44 | 2026-03-24 10:15:00 | 122.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 131.90 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2026-03-20 13:00:00 | 131.90 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2026-03-20 13:45:00 | 131.71 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2026-03-23 09:15:00 | 128.44 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-04-02 09:15:00 | 119.11 | 2026-04-02 14:15:00 | 123.38 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-04-02 14:30:00 | 121.79 | 2026-04-02 15:15:00 | 123.50 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-20 10:30:00 | 167.30 | 2026-04-20 15:15:00 | 164.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-04-29 13:45:00 | 162.39 | 2026-05-05 11:15:00 | 162.17 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-05-05 10:00:00 | 162.30 | 2026-05-05 11:15:00 | 162.17 | STOP_HIT | 1.00 | 0.08% |
