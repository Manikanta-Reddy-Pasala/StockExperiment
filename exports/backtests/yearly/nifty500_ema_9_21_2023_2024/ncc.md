# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 170.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 226 |
| ALERT1 | 153 |
| ALERT2 | 151 |
| ALERT2_SKIP | 96 |
| ALERT3 | 350 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 129 |
| PARTIAL | 19 |
| TARGET_HIT | 2 |
| STOP_HIT | 130 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 104
- **Target hits / Stop hits / Partials:** 2 / 130 / 19
- **Avg / median % per leg:** -0.04% / -1.36%
- **Sum % (uncompounded):** -6.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 9 | 12.3% | 1 | 71 | 1 | -1.42% | -103.9% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.06% | 9.2% |
| BUY @ 3rd Alert (retest2) | 70 | 6 | 8.6% | 1 | 69 | 0 | -1.62% | -113.1% |
| SELL (all) | 78 | 38 | 48.7% | 1 | 59 | 18 | 1.25% | 97.3% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.47% | 6.9% |
| SELL @ 3rd Alert (retest2) | 76 | 36 | 47.4% | 1 | 58 | 17 | 1.19% | 90.4% |
| retest1 (combined) | 5 | 5 | 100.0% | 0 | 3 | 2 | 3.22% | 16.1% |
| retest2 (combined) | 146 | 42 | 28.8% | 2 | 127 | 17 | -0.16% | -22.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 10:15:00 | 115.80 | 117.40 | 117.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 12:15:00 | 115.50 | 116.78 | 117.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 113.85 | 113.60 | 114.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 114.05 | 113.40 | 114.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 114.05 | 113.40 | 114.04 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 113.70 | 108.59 | 108.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 11:15:00 | 115.85 | 110.71 | 109.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 114.35 | 114.64 | 112.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 15:15:00 | 113.90 | 114.10 | 112.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 113.90 | 114.10 | 112.98 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 12:15:00 | 123.45 | 124.71 | 124.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 13:15:00 | 123.15 | 124.40 | 124.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-08 10:15:00 | 123.90 | 123.62 | 124.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 123.15 | 123.53 | 124.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 123.15 | 123.53 | 124.02 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 125.25 | 122.97 | 122.85 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 122.00 | 123.13 | 123.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 14:15:00 | 121.00 | 121.94 | 122.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 09:15:00 | 121.55 | 120.57 | 121.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 121.55 | 120.57 | 121.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 121.55 | 120.57 | 121.14 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 122.35 | 121.59 | 121.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 125.05 | 122.38 | 121.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 12:15:00 | 122.80 | 122.80 | 122.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 14:15:00 | 122.20 | 122.65 | 122.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 122.20 | 122.65 | 122.25 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 121.00 | 122.25 | 122.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 11:15:00 | 120.00 | 121.52 | 121.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 121.85 | 119.75 | 120.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 121.85 | 119.75 | 120.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 121.85 | 119.75 | 120.25 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 120.90 | 120.58 | 120.56 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 11:15:00 | 120.15 | 120.54 | 120.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 12:15:00 | 119.60 | 120.35 | 120.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 14:15:00 | 120.90 | 120.35 | 120.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 120.90 | 120.35 | 120.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 120.90 | 120.35 | 120.45 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 120.95 | 120.58 | 120.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 13:15:00 | 122.15 | 120.91 | 120.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 15:15:00 | 121.60 | 121.78 | 121.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 121.05 | 121.64 | 121.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 121.05 | 121.64 | 121.40 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 13:15:00 | 121.10 | 121.36 | 121.39 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 14:15:00 | 122.15 | 121.52 | 121.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 15:15:00 | 123.70 | 121.96 | 121.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 127.25 | 127.51 | 125.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 126.15 | 127.08 | 126.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 126.15 | 127.08 | 126.20 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 137.60 | 139.44 | 139.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 10:15:00 | 136.25 | 138.45 | 139.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 14:15:00 | 137.55 | 137.45 | 138.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 139.60 | 137.87 | 138.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 139.60 | 137.87 | 138.38 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 140.15 | 138.70 | 138.69 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 15:15:00 | 138.25 | 139.07 | 139.11 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 141.90 | 139.63 | 139.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 10:15:00 | 146.80 | 141.07 | 140.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 14:15:00 | 155.10 | 155.20 | 152.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 153.15 | 155.13 | 152.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 153.15 | 155.13 | 152.90 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 13:15:00 | 150.25 | 152.04 | 152.25 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 153.35 | 152.52 | 152.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 12:15:00 | 153.70 | 152.76 | 152.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 15:15:00 | 152.85 | 152.99 | 152.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 15:15:00 | 152.85 | 152.99 | 152.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 152.85 | 152.99 | 152.73 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 152.55 | 153.89 | 154.00 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 11:15:00 | 158.60 | 154.72 | 154.26 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 151.80 | 154.95 | 155.18 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 157.80 | 155.13 | 155.01 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 152.85 | 155.42 | 155.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 152.75 | 154.53 | 155.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 09:15:00 | 153.20 | 151.56 | 152.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 09:15:00 | 153.20 | 151.56 | 152.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 153.20 | 151.56 | 152.63 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 11:15:00 | 153.90 | 152.82 | 152.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 13:15:00 | 155.10 | 153.80 | 153.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 153.65 | 154.13 | 153.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 153.65 | 154.13 | 153.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 153.65 | 154.13 | 153.68 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 152.40 | 153.24 | 153.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 151.95 | 152.98 | 153.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 152.10 | 152.07 | 152.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 153.75 | 152.24 | 152.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 153.75 | 152.24 | 152.52 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 154.35 | 152.82 | 152.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 161.15 | 155.09 | 153.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 15:15:00 | 169.20 | 169.54 | 165.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 13:15:00 | 166.20 | 169.04 | 166.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 166.20 | 169.04 | 166.71 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 164.50 | 167.29 | 167.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 14:15:00 | 162.45 | 166.32 | 167.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 12:15:00 | 161.10 | 159.81 | 161.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 12:15:00 | 161.10 | 159.81 | 161.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 161.10 | 159.81 | 161.70 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 12:15:00 | 167.00 | 161.76 | 161.72 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 156.50 | 161.95 | 161.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 153.85 | 158.65 | 160.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 152.55 | 150.44 | 153.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 152.55 | 150.44 | 153.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 152.55 | 150.44 | 153.43 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 10:15:00 | 152.30 | 149.60 | 149.36 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 15:15:00 | 149.20 | 150.41 | 150.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 147.40 | 149.81 | 150.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 150.75 | 149.30 | 149.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 150.75 | 149.30 | 149.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 150.75 | 149.30 | 149.80 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 152.85 | 150.09 | 149.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 10:15:00 | 154.25 | 150.92 | 150.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 14:15:00 | 157.35 | 158.07 | 155.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 155.90 | 157.71 | 156.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 155.90 | 157.71 | 156.74 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 14:15:00 | 154.60 | 156.64 | 156.64 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 158.90 | 156.83 | 156.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 159.90 | 157.44 | 157.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 158.70 | 159.24 | 158.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 10:15:00 | 160.35 | 159.46 | 158.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 160.35 | 159.46 | 158.46 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 15:15:00 | 157.05 | 157.84 | 157.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 11:15:00 | 156.45 | 157.45 | 157.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 158.55 | 157.42 | 157.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 158.55 | 157.42 | 157.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 158.55 | 157.42 | 157.57 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 11:15:00 | 158.30 | 157.76 | 157.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 12:15:00 | 159.30 | 158.07 | 157.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 15:15:00 | 158.50 | 158.53 | 158.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 155.85 | 157.99 | 157.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 155.85 | 157.99 | 157.95 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 155.15 | 157.43 | 157.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 154.65 | 156.51 | 157.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 158.50 | 155.79 | 156.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 158.50 | 155.79 | 156.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 158.50 | 155.79 | 156.53 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 157.05 | 156.94 | 156.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 158.35 | 157.54 | 157.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 11:15:00 | 156.95 | 157.42 | 157.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 11:15:00 | 156.95 | 157.42 | 157.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 156.95 | 157.42 | 157.23 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 13:15:00 | 156.10 | 156.97 | 157.04 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 09:15:00 | 164.50 | 158.27 | 157.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 14:15:00 | 165.80 | 162.59 | 160.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 162.95 | 162.96 | 160.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 15:15:00 | 161.90 | 162.57 | 161.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 161.90 | 162.57 | 161.56 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 159.30 | 161.93 | 162.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 158.25 | 160.66 | 161.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 15:15:00 | 159.30 | 159.19 | 160.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 158.70 | 159.09 | 160.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 158.70 | 159.09 | 160.00 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 147.40 | 146.49 | 146.41 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 13:15:00 | 144.05 | 146.08 | 146.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 141.95 | 144.08 | 144.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 144.30 | 143.10 | 144.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 144.30 | 143.10 | 144.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 144.30 | 143.10 | 144.11 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 146.40 | 144.91 | 144.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 148.40 | 145.86 | 145.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 10:15:00 | 151.15 | 151.31 | 149.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 13:15:00 | 149.35 | 150.66 | 149.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 149.35 | 150.66 | 149.82 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 15:15:00 | 162.00 | 163.20 | 163.24 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 164.75 | 163.51 | 163.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 11:15:00 | 166.10 | 164.21 | 163.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 14:15:00 | 168.55 | 168.77 | 167.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 165.95 | 168.21 | 167.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 165.95 | 168.21 | 167.17 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 162.85 | 166.47 | 166.52 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 165.40 | 164.57 | 164.47 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 12:15:00 | 163.20 | 164.29 | 164.43 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 165.90 | 164.60 | 164.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 166.85 | 165.32 | 164.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 165.50 | 166.33 | 165.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 14:15:00 | 165.50 | 166.33 | 165.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 165.50 | 166.33 | 165.80 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 173.40 | 174.44 | 174.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 171.95 | 173.66 | 174.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 174.65 | 173.06 | 173.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 174.65 | 173.06 | 173.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 174.65 | 173.06 | 173.56 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 175.10 | 173.93 | 173.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 176.45 | 174.55 | 174.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 13:15:00 | 174.10 | 174.91 | 174.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 13:15:00 | 174.10 | 174.91 | 174.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 174.10 | 174.91 | 174.53 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 172.40 | 174.10 | 174.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 10:15:00 | 170.95 | 173.29 | 173.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 170.00 | 168.81 | 170.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 170.00 | 168.81 | 170.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 170.00 | 168.81 | 170.13 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 12:15:00 | 166.20 | 164.99 | 164.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 167.25 | 165.64 | 165.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 167.20 | 167.84 | 167.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 10:15:00 | 169.05 | 168.08 | 167.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 169.05 | 168.08 | 167.21 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 10:15:00 | 166.55 | 167.08 | 167.12 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 09:15:00 | 168.00 | 167.04 | 167.04 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 166.50 | 167.17 | 167.20 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 09:15:00 | 171.65 | 167.98 | 167.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 176.00 | 171.11 | 169.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 174.80 | 174.95 | 172.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 12:15:00 | 174.60 | 174.73 | 173.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 12:15:00 | 174.60 | 174.73 | 173.09 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 13:15:00 | 193.50 | 195.97 | 195.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 189.95 | 193.94 | 194.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 194.65 | 194.08 | 194.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 194.65 | 194.08 | 194.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 194.65 | 194.08 | 194.95 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 14:15:00 | 197.15 | 195.51 | 195.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 200.55 | 196.72 | 196.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 15:15:00 | 203.75 | 204.16 | 201.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 199.05 | 203.14 | 201.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 199.05 | 203.14 | 201.66 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 195.35 | 200.73 | 200.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 193.40 | 197.82 | 199.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 195.90 | 195.47 | 197.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 196.60 | 195.70 | 197.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 196.60 | 195.70 | 197.16 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 203.80 | 198.06 | 197.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 209.70 | 200.39 | 198.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 210.70 | 213.23 | 211.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 11:15:00 | 210.70 | 213.23 | 211.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 210.70 | 213.23 | 211.19 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 216.85 | 218.39 | 218.39 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 14:15:00 | 218.80 | 218.47 | 218.43 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 15:15:00 | 218.00 | 218.38 | 218.39 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 09:15:00 | 227.00 | 220.10 | 219.17 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 213.60 | 220.00 | 220.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 211.30 | 218.26 | 219.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 15:15:00 | 216.00 | 215.17 | 217.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 215.95 | 215.32 | 217.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 215.95 | 215.32 | 217.10 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 09:15:00 | 222.10 | 218.05 | 217.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 227.10 | 221.86 | 220.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 13:15:00 | 222.70 | 223.12 | 221.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 223.15 | 224.56 | 223.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 223.15 | 224.56 | 223.44 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 248.15 | 252.90 | 253.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 247.55 | 251.83 | 252.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 13:15:00 | 248.00 | 247.06 | 249.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 247.00 | 247.03 | 249.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 247.00 | 247.03 | 249.12 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 254.35 | 250.16 | 250.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 256.80 | 251.49 | 250.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 10:15:00 | 253.05 | 253.74 | 252.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 11:15:00 | 251.30 | 253.25 | 252.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 251.30 | 253.25 | 252.36 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 15:15:00 | 250.90 | 251.72 | 251.81 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 09:15:00 | 253.75 | 252.13 | 251.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 10:15:00 | 255.60 | 252.82 | 252.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 255.90 | 257.43 | 255.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 255.90 | 257.43 | 255.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 255.90 | 257.43 | 255.23 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 245.00 | 254.45 | 255.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 243.65 | 249.20 | 252.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 222.00 | 219.82 | 228.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 238.60 | 224.86 | 228.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 238.60 | 224.86 | 228.14 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 233.10 | 229.83 | 229.61 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 225.55 | 229.52 | 229.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 10:15:00 | 222.10 | 228.04 | 228.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 218.55 | 217.57 | 221.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 218.55 | 217.57 | 221.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 218.55 | 217.57 | 221.20 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 233.65 | 223.07 | 222.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 236.65 | 228.75 | 225.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 234.05 | 234.62 | 231.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 234.05 | 234.62 | 231.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 234.05 | 234.62 | 231.47 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 231.95 | 237.67 | 237.86 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 239.90 | 238.13 | 237.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 10:15:00 | 245.95 | 242.44 | 240.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 13:15:00 | 256.95 | 257.04 | 251.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 09:15:00 | 249.95 | 255.69 | 252.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 249.95 | 255.69 | 252.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 263.40 | 265.37 | 264.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:45:00 | 262.70 | 264.52 | 264.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 10:15:00 | 260.45 | 263.71 | 264.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 10:15:00 | 260.45 | 263.71 | 264.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 11:15:00 | 259.70 | 262.90 | 263.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 15:15:00 | 243.50 | 243.43 | 246.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-22 09:15:00 | 242.85 | 243.43 | 246.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 243.00 | 243.34 | 246.03 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 14:15:00 | 246.20 | 245.61 | 245.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 252.55 | 247.15 | 246.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 249.40 | 250.69 | 248.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 15:00:00 | 249.40 | 250.69 | 248.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 249.20 | 250.40 | 248.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:15:00 | 248.05 | 250.40 | 248.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 248.15 | 249.95 | 248.70 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 14:15:00 | 246.60 | 247.98 | 248.07 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 250.35 | 248.36 | 248.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 11:15:00 | 254.15 | 249.86 | 248.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 249.60 | 249.93 | 249.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 249.60 | 249.93 | 249.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 247.70 | 249.49 | 249.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 247.70 | 249.49 | 249.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 248.75 | 249.34 | 249.11 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 12:15:00 | 247.70 | 248.86 | 248.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 13:15:00 | 247.05 | 248.50 | 248.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 11:15:00 | 245.00 | 244.04 | 245.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-02 12:00:00 | 245.00 | 244.04 | 245.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 246.10 | 244.45 | 245.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:00:00 | 246.10 | 244.45 | 245.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 246.40 | 244.84 | 245.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:45:00 | 246.90 | 244.84 | 245.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 246.70 | 245.21 | 245.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 15:00:00 | 246.70 | 245.21 | 245.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 244.80 | 245.72 | 245.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:45:00 | 242.80 | 245.30 | 245.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 11:15:00 | 247.20 | 245.79 | 245.82 | SL hit (close>static) qty=1.00 sl=246.40 alert=retest2 |

### Cycle 84 — BUY (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 12:15:00 | 251.00 | 246.83 | 246.29 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 13:15:00 | 244.80 | 246.77 | 246.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 14:15:00 | 241.40 | 245.69 | 246.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 11:15:00 | 247.20 | 245.36 | 245.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 11:15:00 | 247.20 | 245.36 | 245.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 247.20 | 245.36 | 245.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 247.20 | 245.36 | 245.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 245.90 | 245.47 | 245.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:30:00 | 245.50 | 245.59 | 245.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:30:00 | 245.65 | 245.74 | 246.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 244.25 | 245.74 | 246.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 244.05 | 245.69 | 245.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 241.55 | 244.86 | 245.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:15:00 | 238.60 | 242.58 | 244.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 233.22 | 239.59 | 242.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 233.37 | 239.59 | 242.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 232.04 | 239.59 | 242.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 231.85 | 239.59 | 242.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 10:15:00 | 240.00 | 239.67 | 242.07 | SL hit (close>ema200) qty=0.50 sl=239.67 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 248.35 | 241.98 | 241.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 253.10 | 247.01 | 244.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 250.80 | 251.92 | 249.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 250.80 | 251.92 | 249.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 286.15 | 277.25 | 271.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:45:00 | 287.80 | 280.31 | 273.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 15:00:00 | 288.05 | 284.48 | 277.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 288.30 | 286.65 | 279.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:45:00 | 287.90 | 286.78 | 281.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 286.10 | 287.63 | 285.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 286.20 | 287.63 | 285.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 285.80 | 287.26 | 285.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 286.00 | 287.26 | 285.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 285.55 | 286.92 | 285.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:45:00 | 285.30 | 286.92 | 285.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 285.40 | 286.62 | 285.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:00:00 | 285.40 | 286.62 | 285.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 285.30 | 286.35 | 285.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:30:00 | 285.55 | 286.35 | 285.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 282.50 | 285.58 | 285.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:30:00 | 282.55 | 285.58 | 285.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-24 15:15:00 | 281.90 | 284.85 | 284.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 281.90 | 284.85 | 284.94 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 285.15 | 284.58 | 284.57 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 284.15 | 284.56 | 284.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 279.65 | 282.93 | 283.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 286.20 | 281.85 | 282.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 286.20 | 281.85 | 282.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 286.20 | 281.85 | 282.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:30:00 | 287.60 | 281.85 | 282.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 289.90 | 283.46 | 283.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 11:15:00 | 294.55 | 287.56 | 286.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 287.25 | 289.29 | 287.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 287.25 | 289.29 | 287.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 287.25 | 289.29 | 287.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 287.25 | 289.29 | 287.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 288.00 | 289.03 | 287.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 297.60 | 289.03 | 287.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 271.60 | 297.39 | 295.80 | SL hit (close<static) qty=1.00 sl=286.25 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 252.45 | 288.40 | 291.86 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 312.05 | 288.59 | 286.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 314.40 | 293.75 | 288.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 318.75 | 320.97 | 311.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 318.75 | 320.97 | 311.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 324.70 | 327.35 | 323.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 327.40 | 327.35 | 323.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 325.90 | 327.06 | 323.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:30:00 | 333.00 | 328.28 | 324.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:45:00 | 329.25 | 328.50 | 327.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 332.00 | 327.23 | 326.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:15:00 | 329.50 | 327.50 | 327.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 328.40 | 327.79 | 327.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:30:00 | 327.65 | 327.79 | 327.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 323.10 | 327.04 | 327.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 323.10 | 327.04 | 327.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 15:15:00 | 322.50 | 324.60 | 325.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 318.60 | 318.08 | 320.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 318.60 | 318.08 | 320.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 318.60 | 318.08 | 320.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 319.00 | 318.08 | 320.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 320.00 | 318.51 | 320.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 320.00 | 318.51 | 320.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 321.05 | 319.02 | 320.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:30:00 | 321.40 | 319.02 | 320.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 327.00 | 320.62 | 321.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:45:00 | 326.55 | 320.62 | 321.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 15:15:00 | 324.90 | 321.47 | 321.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 329.25 | 323.03 | 322.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 324.10 | 324.49 | 323.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 324.10 | 324.49 | 323.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 323.80 | 324.35 | 323.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 323.40 | 324.35 | 323.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 328.65 | 325.21 | 323.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:30:00 | 333.90 | 326.31 | 325.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 331.60 | 326.31 | 325.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:00:00 | 331.70 | 328.02 | 326.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 324.35 | 325.66 | 325.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 324.35 | 325.66 | 325.83 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 329.25 | 325.99 | 325.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 330.60 | 327.31 | 326.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 11:15:00 | 325.65 | 327.22 | 326.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 325.65 | 327.22 | 326.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 325.65 | 327.22 | 326.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 325.65 | 327.22 | 326.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 323.00 | 326.37 | 326.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 323.00 | 326.37 | 326.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 319.60 | 325.02 | 325.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 14:15:00 | 316.05 | 323.23 | 324.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 321.25 | 320.87 | 322.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 13:15:00 | 321.25 | 320.87 | 322.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 321.25 | 320.87 | 322.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:00:00 | 321.25 | 320.87 | 322.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 315.55 | 319.80 | 322.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:45:00 | 321.25 | 319.80 | 322.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 320.20 | 319.34 | 321.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 319.70 | 319.34 | 321.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 320.70 | 319.61 | 321.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 321.05 | 319.61 | 321.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 320.75 | 319.84 | 321.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:00:00 | 320.75 | 319.84 | 321.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 321.85 | 320.25 | 321.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:00:00 | 321.85 | 320.25 | 321.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 320.00 | 320.20 | 321.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:15:00 | 321.65 | 320.20 | 321.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 321.65 | 320.49 | 321.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 331.60 | 320.49 | 321.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 332.60 | 322.91 | 322.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 345.20 | 331.97 | 327.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 336.70 | 338.13 | 333.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:30:00 | 337.20 | 338.13 | 333.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 336.35 | 338.74 | 336.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 335.10 | 338.74 | 336.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 335.75 | 338.14 | 336.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 336.40 | 338.14 | 336.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 337.30 | 337.97 | 336.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 341.10 | 336.84 | 336.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 334.95 | 336.50 | 336.44 | SL hit (close<static) qty=1.00 sl=335.50 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 335.20 | 336.42 | 336.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 329.45 | 335.03 | 335.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 331.00 | 330.77 | 332.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 330.30 | 330.77 | 332.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 328.85 | 330.39 | 332.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 327.65 | 330.39 | 332.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 327.50 | 329.24 | 331.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 13:15:00 | 333.55 | 330.16 | 331.58 | SL hit (close>static) qty=1.00 sl=332.55 alert=retest2 |

### Cycle 100 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 333.55 | 331.03 | 330.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 334.70 | 332.71 | 331.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 331.85 | 332.54 | 331.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 11:15:00 | 331.85 | 332.54 | 331.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 331.85 | 332.54 | 331.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 331.85 | 332.54 | 331.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 332.20 | 332.47 | 331.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 332.20 | 332.47 | 331.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 330.85 | 332.15 | 331.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 330.85 | 332.15 | 331.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 329.10 | 331.54 | 331.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 326.50 | 330.28 | 330.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 315.60 | 313.63 | 318.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 315.60 | 313.63 | 318.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 315.50 | 314.00 | 318.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 316.85 | 314.00 | 318.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 318.40 | 315.29 | 318.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 318.40 | 315.29 | 318.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 318.50 | 315.93 | 318.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 318.60 | 315.93 | 318.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 316.45 | 316.03 | 317.93 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 328.55 | 319.50 | 318.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 341.15 | 323.59 | 320.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 333.25 | 336.10 | 331.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 333.25 | 336.10 | 331.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 333.25 | 336.10 | 331.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 333.25 | 336.10 | 331.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 332.15 | 334.82 | 331.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:30:00 | 331.60 | 334.82 | 331.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 332.85 | 334.43 | 331.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 329.95 | 334.43 | 331.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 340.60 | 338.57 | 336.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:30:00 | 338.80 | 338.57 | 336.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 339.70 | 340.69 | 338.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 339.70 | 340.69 | 338.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 339.30 | 340.41 | 338.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:30:00 | 343.75 | 342.02 | 340.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:00:00 | 342.70 | 348.40 | 347.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 343.85 | 347.28 | 347.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 343.85 | 347.28 | 347.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 325.65 | 341.56 | 344.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 326.20 | 321.35 | 327.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 09:45:00 | 322.65 | 321.35 | 327.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 325.80 | 322.24 | 327.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:15:00 | 322.55 | 323.67 | 326.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:30:00 | 323.00 | 322.89 | 326.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 324.50 | 321.11 | 322.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 324.20 | 321.73 | 322.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 15:15:00 | 324.65 | 323.38 | 323.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 15:15:00 | 324.65 | 323.38 | 323.31 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 321.00 | 322.91 | 323.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 318.15 | 321.10 | 321.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 324.30 | 313.84 | 315.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 324.30 | 313.84 | 315.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 324.30 | 313.84 | 315.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 324.30 | 313.84 | 315.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 324.80 | 316.03 | 316.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 327.10 | 322.02 | 319.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 322.35 | 323.15 | 320.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 323.60 | 323.15 | 320.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 320.10 | 322.19 | 320.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:30:00 | 319.35 | 322.19 | 320.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 318.20 | 321.39 | 320.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 318.20 | 321.39 | 320.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 317.85 | 320.68 | 320.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:30:00 | 317.55 | 320.68 | 320.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 320.60 | 321.08 | 320.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:00:00 | 320.60 | 321.08 | 320.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 320.00 | 320.87 | 320.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 321.00 | 320.72 | 320.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:45:00 | 323.40 | 321.09 | 320.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 14:30:00 | 321.50 | 321.27 | 320.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:00:00 | 321.25 | 321.27 | 320.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 320.65 | 321.15 | 320.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 320.60 | 321.15 | 320.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 322.80 | 321.48 | 321.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:30:00 | 324.75 | 322.30 | 321.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 324.80 | 322.94 | 322.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:00:00 | 325.80 | 323.03 | 322.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 11:15:00 | 324.65 | 323.28 | 322.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 325.00 | 325.08 | 324.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 325.60 | 325.08 | 324.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 325.00 | 324.99 | 324.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:45:00 | 323.85 | 324.99 | 324.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 323.30 | 324.68 | 324.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:30:00 | 323.35 | 324.68 | 324.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 323.85 | 324.51 | 324.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 324.50 | 324.51 | 324.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 323.80 | 324.37 | 324.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 321.35 | 323.76 | 323.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 321.35 | 323.76 | 323.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 320.95 | 323.20 | 323.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 322.15 | 321.80 | 322.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 322.15 | 321.80 | 322.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 322.15 | 321.80 | 322.68 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 325.85 | 322.45 | 322.38 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 316.00 | 323.18 | 323.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 312.80 | 317.23 | 320.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 314.05 | 312.76 | 315.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 13:15:00 | 316.00 | 313.86 | 315.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 316.00 | 313.86 | 315.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 316.35 | 313.86 | 315.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 316.90 | 314.47 | 315.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:30:00 | 317.45 | 314.47 | 315.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 317.60 | 315.10 | 315.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:30:00 | 319.00 | 315.75 | 316.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 318.25 | 316.25 | 316.22 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 312.10 | 315.40 | 315.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 310.60 | 314.44 | 315.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 316.00 | 314.23 | 315.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 316.00 | 314.23 | 315.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 316.00 | 314.23 | 315.08 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 316.70 | 315.57 | 315.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 317.70 | 316.00 | 315.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 317.05 | 317.36 | 316.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 317.05 | 317.36 | 316.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 317.05 | 317.36 | 316.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:30:00 | 316.50 | 317.36 | 316.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 316.65 | 317.22 | 316.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 314.05 | 317.22 | 316.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 312.70 | 316.32 | 316.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 310.70 | 314.58 | 315.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 315.15 | 312.67 | 313.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 315.15 | 312.67 | 313.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 315.15 | 312.67 | 313.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:45:00 | 315.10 | 312.67 | 313.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 315.25 | 313.19 | 313.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 315.25 | 313.19 | 313.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 315.05 | 313.82 | 313.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 315.05 | 313.82 | 313.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 313.25 | 313.71 | 313.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:15:00 | 313.10 | 313.71 | 313.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:45:00 | 312.75 | 313.34 | 313.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:45:00 | 311.40 | 312.06 | 312.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 315.85 | 309.80 | 309.94 | SL hit (close>static) qty=1.00 sl=315.60 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 314.75 | 310.79 | 310.38 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 310.70 | 312.19 | 312.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 307.30 | 311.11 | 311.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 306.45 | 306.11 | 308.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 306.45 | 306.11 | 308.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 308.20 | 306.67 | 307.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 302.35 | 306.31 | 307.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:00:00 | 303.40 | 303.30 | 305.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:00:00 | 304.10 | 303.46 | 305.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:00:00 | 302.75 | 304.17 | 304.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 305.25 | 304.39 | 304.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 305.00 | 304.39 | 304.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 305.05 | 304.52 | 304.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 305.20 | 304.52 | 304.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 304.55 | 304.53 | 304.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 304.55 | 304.53 | 304.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 305.10 | 304.59 | 304.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:45:00 | 306.10 | 304.59 | 304.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 307.00 | 305.07 | 305.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 10:15:00 | 307.00 | 305.07 | 305.03 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 303.40 | 304.74 | 304.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 300.70 | 303.93 | 304.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 301.60 | 301.58 | 302.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 301.60 | 301.58 | 302.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 296.25 | 300.03 | 301.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 293.45 | 298.72 | 300.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 295.65 | 297.92 | 300.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 293.85 | 295.93 | 298.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 295.60 | 295.93 | 298.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 295.65 | 296.10 | 297.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 297.00 | 296.10 | 297.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 297.50 | 296.49 | 297.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 297.50 | 296.49 | 297.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 300.40 | 297.27 | 297.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 300.40 | 297.27 | 297.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 302.00 | 298.22 | 298.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 302.65 | 298.22 | 298.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 302.00 | 298.97 | 298.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 302.00 | 298.97 | 298.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 306.95 | 301.31 | 300.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 300.80 | 304.01 | 302.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 14:15:00 | 300.80 | 304.01 | 302.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 300.80 | 304.01 | 302.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 300.80 | 304.01 | 302.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 302.00 | 303.61 | 302.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 300.00 | 303.61 | 302.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 300.80 | 303.05 | 302.14 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 298.95 | 301.18 | 301.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 297.75 | 300.50 | 301.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 300.15 | 299.80 | 300.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 300.15 | 299.80 | 300.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 300.15 | 299.80 | 300.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 296.90 | 299.05 | 300.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 301.90 | 299.22 | 299.41 | SL hit (close>static) qty=1.00 sl=301.30 alert=retest2 |

### Cycle 120 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 300.90 | 299.76 | 299.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 305.85 | 301.00 | 300.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 305.75 | 306.33 | 303.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 305.75 | 306.33 | 303.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 309.00 | 308.89 | 306.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:30:00 | 311.45 | 309.47 | 307.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:30:00 | 310.95 | 309.79 | 307.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 13:00:00 | 311.05 | 309.79 | 307.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 14:15:00 | 302.35 | 307.04 | 307.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 302.35 | 307.04 | 307.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 302.00 | 306.03 | 307.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 289.10 | 287.92 | 291.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 289.10 | 287.92 | 291.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 281.95 | 280.24 | 282.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 284.70 | 280.24 | 282.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 284.00 | 280.99 | 282.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 284.85 | 281.86 | 283.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 285.35 | 282.56 | 283.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:30:00 | 285.40 | 282.56 | 283.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 286.00 | 283.92 | 283.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 287.50 | 284.64 | 284.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 292.00 | 292.06 | 289.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:15:00 | 297.55 | 292.06 | 289.44 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-01 17:15:00 | 312.43 | 300.38 | 295.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-04 15:15:00 | 307.65 | 307.95 | 303.16 | SL hit (close<ema200) qty=0.50 sl=307.95 alert=retest1 |

### Cycle 123 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 303.20 | 307.08 | 307.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 299.25 | 304.12 | 305.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 283.00 | 279.23 | 282.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 283.00 | 279.23 | 282.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 283.00 | 279.23 | 282.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 276.55 | 280.54 | 281.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:00:00 | 276.10 | 278.13 | 280.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 284.15 | 279.26 | 279.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 284.15 | 279.26 | 279.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 295.45 | 283.72 | 281.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 295.75 | 295.77 | 290.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:00:00 | 295.75 | 295.77 | 290.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 307.55 | 311.66 | 310.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 307.55 | 311.66 | 310.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 306.00 | 310.53 | 310.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 306.00 | 310.53 | 310.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 306.60 | 309.74 | 309.97 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 313.35 | 310.67 | 310.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 316.50 | 312.31 | 311.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 314.40 | 314.44 | 312.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 15:00:00 | 314.40 | 314.44 | 312.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 313.30 | 314.12 | 313.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 313.30 | 314.12 | 313.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 313.75 | 314.05 | 313.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 315.30 | 314.10 | 313.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:15:00 | 315.25 | 314.10 | 313.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 13:00:00 | 314.80 | 314.24 | 313.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 13:45:00 | 314.85 | 314.17 | 313.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 313.20 | 313.98 | 313.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 313.20 | 313.98 | 313.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 313.00 | 313.78 | 313.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 311.90 | 313.78 | 313.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 309.35 | 312.90 | 312.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 309.35 | 312.90 | 312.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 12:15:00 | 307.50 | 310.34 | 311.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 302.80 | 301.51 | 304.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 302.80 | 301.51 | 304.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 303.60 | 302.35 | 303.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 301.60 | 302.10 | 303.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 308.20 | 303.01 | 303.65 | SL hit (close>static) qty=1.00 sl=304.50 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 309.10 | 304.23 | 304.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 314.00 | 306.95 | 305.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 306.65 | 309.79 | 308.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 306.65 | 309.79 | 308.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 306.65 | 309.79 | 308.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 306.65 | 309.79 | 308.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 308.65 | 309.56 | 308.12 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 303.95 | 307.37 | 307.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 14:15:00 | 302.40 | 306.38 | 306.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 286.00 | 285.74 | 289.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 11:00:00 | 286.00 | 285.74 | 289.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 284.85 | 283.70 | 285.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 284.85 | 283.70 | 285.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 284.50 | 283.89 | 285.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 284.00 | 283.89 | 285.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 283.30 | 283.80 | 285.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 15:15:00 | 269.80 | 274.45 | 278.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 15:15:00 | 269.13 | 274.45 | 278.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 273.50 | 273.47 | 277.22 | SL hit (close>ema200) qty=0.50 sl=273.47 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 278.45 | 276.84 | 276.64 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 270.70 | 276.05 | 276.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 264.80 | 273.80 | 275.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 270.00 | 266.57 | 270.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 270.00 | 266.57 | 270.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 270.00 | 266.57 | 270.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 267.45 | 266.57 | 270.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 270.75 | 267.41 | 270.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 270.75 | 267.41 | 270.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 271.05 | 268.13 | 270.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 271.80 | 268.13 | 270.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 274.50 | 270.19 | 271.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 274.45 | 270.19 | 271.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 247.00 | 244.88 | 247.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 245.15 | 244.88 | 247.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 242.75 | 244.45 | 247.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:30:00 | 241.00 | 243.62 | 245.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 15:15:00 | 247.75 | 246.17 | 246.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 15:15:00 | 247.75 | 246.17 | 246.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 252.90 | 247.51 | 246.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 14:15:00 | 250.25 | 250.79 | 249.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 15:00:00 | 250.25 | 250.79 | 249.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 249.90 | 250.61 | 249.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 251.15 | 250.61 | 249.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 247.20 | 249.93 | 249.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 247.20 | 249.93 | 249.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 243.70 | 248.68 | 248.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 242.80 | 247.08 | 248.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 243.95 | 241.38 | 243.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 243.95 | 241.38 | 243.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 243.95 | 241.38 | 243.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 243.95 | 241.38 | 243.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 243.20 | 241.75 | 243.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 242.15 | 241.75 | 243.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 244.35 | 242.27 | 243.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:30:00 | 244.35 | 242.27 | 243.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 245.70 | 242.95 | 243.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 246.20 | 242.95 | 243.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 246.45 | 244.13 | 244.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 246.30 | 244.13 | 244.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 246.75 | 244.66 | 244.58 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 239.85 | 244.36 | 244.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 11:15:00 | 239.20 | 242.60 | 243.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 229.35 | 227.55 | 232.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 229.35 | 227.55 | 232.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 234.95 | 228.26 | 230.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 235.45 | 228.26 | 230.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 238.20 | 230.25 | 231.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 238.20 | 230.25 | 231.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 236.65 | 232.77 | 232.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 244.90 | 236.81 | 234.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 234.00 | 248.72 | 245.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 234.00 | 248.72 | 245.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 234.00 | 248.72 | 245.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 234.00 | 248.72 | 245.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 238.10 | 246.60 | 245.16 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 231.70 | 243.62 | 243.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 221.85 | 237.39 | 240.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 227.15 | 222.81 | 229.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:00:00 | 227.15 | 222.81 | 229.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 229.35 | 224.57 | 229.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 229.35 | 224.57 | 229.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 230.55 | 225.77 | 229.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:45:00 | 231.15 | 225.77 | 229.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 230.90 | 226.79 | 229.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 231.05 | 226.79 | 229.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 246.50 | 232.55 | 231.72 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 211.00 | 231.75 | 233.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 199.85 | 207.31 | 214.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 201.00 | 199.01 | 205.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 201.00 | 199.01 | 205.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 205.05 | 200.22 | 205.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 205.90 | 200.22 | 205.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 202.85 | 200.75 | 205.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 203.75 | 200.75 | 205.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 202.30 | 200.67 | 203.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 203.10 | 200.67 | 203.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 203.10 | 201.16 | 203.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 202.95 | 201.16 | 203.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 190.90 | 187.21 | 189.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 190.90 | 187.21 | 189.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 189.80 | 187.72 | 189.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 191.15 | 187.72 | 189.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 192.20 | 188.62 | 189.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:45:00 | 191.95 | 188.62 | 189.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 189.55 | 189.04 | 189.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:30:00 | 189.50 | 189.04 | 189.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 189.15 | 189.06 | 189.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:15:00 | 190.35 | 189.06 | 189.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 190.35 | 189.32 | 189.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 190.70 | 189.32 | 189.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 194.70 | 190.40 | 190.12 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 188.05 | 190.75 | 190.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 14:15:00 | 186.85 | 189.97 | 190.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 186.30 | 185.94 | 187.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 186.30 | 185.94 | 187.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 186.30 | 185.94 | 187.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 13:00:00 | 182.65 | 184.70 | 186.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 12:15:00 | 173.52 | 176.87 | 179.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 177.59 | 174.75 | 177.02 | SL hit (close>ema200) qty=0.50 sl=174.75 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 182.50 | 178.15 | 177.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 184.52 | 181.01 | 179.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 186.18 | 186.46 | 184.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 187.15 | 186.46 | 184.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 184.36 | 185.87 | 184.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 184.36 | 185.87 | 184.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 184.70 | 185.64 | 184.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:45:00 | 185.92 | 185.39 | 184.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:45:00 | 186.44 | 185.02 | 184.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 12:30:00 | 186.07 | 185.05 | 184.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 183.00 | 184.50 | 184.44 | SL hit (close<static) qty=1.00 sl=183.85 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 183.01 | 184.20 | 184.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 181.29 | 183.62 | 184.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 184.84 | 183.86 | 184.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 184.84 | 183.86 | 184.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 184.84 | 183.86 | 184.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 184.84 | 183.86 | 184.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 183.77 | 183.85 | 184.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:15:00 | 182.77 | 183.85 | 184.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 15:15:00 | 185.37 | 184.39 | 184.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 185.37 | 184.39 | 184.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 09:15:00 | 186.19 | 184.75 | 184.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 184.70 | 185.06 | 184.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 184.70 | 185.06 | 184.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 184.70 | 185.06 | 184.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:45:00 | 184.70 | 185.06 | 184.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 185.83 | 185.21 | 184.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:00:00 | 186.82 | 185.69 | 185.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 187.83 | 186.64 | 186.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 184.35 | 185.68 | 185.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 14:15:00 | 184.35 | 185.68 | 185.80 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 188.89 | 186.10 | 185.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 190.48 | 187.38 | 186.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 200.17 | 200.29 | 196.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:00:00 | 200.17 | 200.29 | 196.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 205.10 | 207.21 | 204.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 205.24 | 207.21 | 204.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 205.06 | 206.78 | 204.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 203.15 | 206.78 | 204.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 203.99 | 206.22 | 204.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 203.99 | 206.22 | 204.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 205.16 | 206.01 | 204.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 214.17 | 205.62 | 204.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 10:15:00 | 208.90 | 211.89 | 212.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 208.90 | 211.89 | 212.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 190.43 | 205.05 | 208.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 202.01 | 198.95 | 202.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 202.01 | 198.95 | 202.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 202.01 | 198.95 | 202.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 202.68 | 198.95 | 202.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 203.50 | 199.86 | 202.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 203.50 | 199.86 | 202.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 204.61 | 200.81 | 202.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 204.61 | 200.81 | 202.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 205.00 | 201.65 | 203.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 205.00 | 201.65 | 203.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 205.05 | 202.62 | 203.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 205.05 | 202.62 | 203.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 205.70 | 203.24 | 203.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 203.82 | 203.24 | 203.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 206.71 | 203.96 | 203.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 206.71 | 203.96 | 203.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 207.10 | 204.96 | 204.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 218.16 | 219.13 | 216.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 218.16 | 219.13 | 216.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 216.69 | 218.39 | 216.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:00:00 | 216.69 | 218.39 | 216.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 219.34 | 218.30 | 217.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 220.24 | 218.64 | 217.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 220.48 | 218.64 | 217.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 216.30 | 219.28 | 219.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 216.30 | 219.28 | 219.43 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 14:15:00 | 219.95 | 219.50 | 219.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 220.03 | 219.61 | 219.53 | Break + close above crossover candle high |

### Cycle 151 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 11:15:00 | 218.72 | 219.43 | 219.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 12:15:00 | 218.21 | 219.19 | 219.34 | Break + close below crossover candle low |

### Cycle 152 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 221.00 | 219.55 | 219.49 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 216.90 | 219.21 | 219.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 213.00 | 217.97 | 218.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 217.00 | 214.73 | 216.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 217.00 | 214.73 | 216.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 217.00 | 214.73 | 216.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 217.00 | 214.73 | 216.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 217.07 | 215.20 | 216.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 217.07 | 215.20 | 216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 217.28 | 215.61 | 216.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 217.23 | 215.61 | 216.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 218.01 | 216.84 | 216.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 221.89 | 216.84 | 216.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 221.31 | 217.73 | 217.31 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 215.77 | 217.12 | 217.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 212.00 | 215.11 | 216.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 216.40 | 214.36 | 215.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 216.40 | 214.36 | 215.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 216.40 | 214.36 | 215.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 213.76 | 214.48 | 215.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:30:00 | 213.72 | 214.54 | 215.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 217.64 | 215.91 | 215.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 217.64 | 215.91 | 215.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 221.31 | 216.99 | 216.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 215.63 | 218.88 | 217.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 215.63 | 218.88 | 217.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 215.63 | 218.88 | 217.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 215.63 | 218.88 | 217.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 215.59 | 218.22 | 217.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:30:00 | 216.08 | 218.22 | 217.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 212.93 | 216.45 | 216.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 211.01 | 215.36 | 216.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 211.98 | 211.53 | 213.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:30:00 | 212.26 | 211.53 | 213.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 212.96 | 211.82 | 213.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 213.83 | 211.82 | 213.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 213.51 | 212.16 | 213.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 213.51 | 212.16 | 213.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 214.14 | 212.55 | 213.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 214.14 | 212.55 | 213.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 214.70 | 212.98 | 213.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 215.29 | 212.98 | 213.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 216.26 | 208.77 | 209.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 216.26 | 208.77 | 209.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 216.10 | 211.54 | 210.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 218.76 | 213.92 | 212.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 225.91 | 228.06 | 225.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 225.91 | 228.06 | 225.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 225.91 | 228.06 | 225.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 229.63 | 227.91 | 225.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 233.20 | 225.58 | 224.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 231.51 | 233.49 | 233.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 231.51 | 233.49 | 233.49 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 234.10 | 233.39 | 233.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 235.15 | 234.29 | 233.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 234.40 | 234.82 | 234.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 13:15:00 | 234.40 | 234.82 | 234.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 234.40 | 234.82 | 234.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 234.40 | 234.82 | 234.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 234.06 | 234.67 | 234.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 235.71 | 234.63 | 234.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 233.15 | 236.14 | 236.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 233.15 | 236.14 | 236.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 232.15 | 235.34 | 235.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 234.75 | 233.32 | 234.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 234.75 | 233.32 | 234.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 234.75 | 233.32 | 234.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 234.75 | 233.32 | 234.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 233.31 | 233.32 | 234.27 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 235.83 | 234.72 | 234.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 236.70 | 235.11 | 234.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 234.88 | 235.25 | 234.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 234.88 | 235.25 | 234.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 234.88 | 235.25 | 234.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 234.88 | 235.25 | 234.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 234.99 | 235.20 | 234.99 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 231.98 | 234.35 | 234.63 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 237.35 | 234.80 | 234.75 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 233.85 | 234.76 | 234.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 233.32 | 234.47 | 234.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 233.58 | 233.57 | 234.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 233.58 | 233.57 | 234.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 233.58 | 233.57 | 234.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 233.58 | 233.57 | 234.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 234.37 | 233.73 | 234.13 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 238.39 | 235.05 | 234.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 240.35 | 237.26 | 235.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 239.49 | 239.54 | 237.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 240.00 | 239.73 | 238.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 239.10 | 239.78 | 238.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 239.10 | 239.78 | 238.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 238.72 | 239.57 | 238.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 238.07 | 239.57 | 238.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 240.02 | 239.66 | 238.90 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 237.80 | 238.50 | 238.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 236.92 | 238.18 | 238.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 227.95 | 227.30 | 230.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 227.41 | 227.30 | 230.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 229.28 | 228.53 | 229.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 230.21 | 228.53 | 229.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 228.92 | 228.61 | 229.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 229.70 | 228.61 | 229.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 222.00 | 221.26 | 222.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 220.60 | 221.26 | 222.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 220.62 | 221.08 | 222.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 225.95 | 222.54 | 222.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 225.95 | 222.54 | 222.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 227.66 | 224.73 | 223.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 224.86 | 224.95 | 223.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 227.20 | 224.95 | 223.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 231.00 | 230.34 | 228.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 231.21 | 230.41 | 229.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 231.15 | 230.38 | 229.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 231.20 | 230.50 | 229.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 231.40 | 230.50 | 229.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 229.00 | 231.44 | 230.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 229.00 | 231.44 | 230.72 | SL hit (close<ema400) qty=1.00 sl=230.72 alert=retest1 |

### Cycle 169 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 227.08 | 229.81 | 230.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 225.76 | 229.00 | 229.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 225.96 | 225.87 | 227.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 225.96 | 225.87 | 227.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 225.96 | 225.87 | 227.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 226.28 | 225.87 | 227.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 223.34 | 223.31 | 224.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 223.54 | 223.31 | 224.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 224.40 | 223.37 | 223.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 224.40 | 223.37 | 223.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 224.10 | 223.52 | 223.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 224.20 | 223.52 | 223.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 224.92 | 223.80 | 224.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 224.92 | 223.80 | 224.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 225.45 | 224.13 | 224.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 225.69 | 224.13 | 224.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 225.99 | 224.50 | 224.37 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 222.60 | 224.35 | 224.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 221.30 | 223.31 | 223.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 222.62 | 222.34 | 223.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 222.62 | 222.34 | 223.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 225.32 | 222.70 | 223.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 225.75 | 222.70 | 223.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 223.30 | 223.16 | 223.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 222.82 | 223.28 | 223.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 225.15 | 223.66 | 223.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 225.15 | 223.66 | 223.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 227.83 | 225.00 | 224.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 228.90 | 229.54 | 228.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:00:00 | 228.90 | 229.54 | 228.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 229.18 | 229.47 | 228.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:45:00 | 228.92 | 229.47 | 228.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 229.71 | 229.40 | 228.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 228.49 | 229.40 | 228.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 227.45 | 229.01 | 228.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 227.45 | 229.01 | 228.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 226.87 | 228.58 | 228.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 227.12 | 228.58 | 228.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 226.01 | 228.07 | 228.10 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 229.20 | 228.15 | 228.08 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 227.49 | 228.10 | 228.11 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 228.60 | 228.20 | 228.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 229.10 | 228.38 | 228.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 227.85 | 228.29 | 228.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 227.85 | 228.29 | 228.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 227.85 | 228.29 | 228.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 227.76 | 228.29 | 228.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 228.12 | 228.26 | 228.22 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 227.71 | 228.12 | 228.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 224.32 | 227.21 | 227.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 13:15:00 | 224.47 | 224.29 | 225.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:00:00 | 224.47 | 224.29 | 225.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 221.00 | 220.59 | 222.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 219.44 | 220.36 | 221.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 222.80 | 221.14 | 221.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 222.80 | 221.14 | 221.01 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 219.77 | 221.41 | 221.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 217.40 | 220.15 | 220.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 217.27 | 212.69 | 214.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 217.27 | 212.69 | 214.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 217.27 | 212.69 | 214.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 218.97 | 212.69 | 214.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 216.55 | 213.46 | 214.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 214.49 | 214.08 | 214.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 215.70 | 214.34 | 214.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 214.77 | 214.66 | 214.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 214.77 | 214.66 | 214.65 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 214.20 | 214.57 | 214.61 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 221.41 | 215.66 | 215.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 224.14 | 217.35 | 215.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 221.64 | 221.77 | 219.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 221.64 | 221.77 | 219.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 222.46 | 222.72 | 221.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 222.10 | 222.72 | 221.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 221.78 | 222.91 | 222.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 221.78 | 222.91 | 222.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 222.49 | 222.83 | 222.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 223.03 | 222.65 | 222.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 218.95 | 221.85 | 221.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 218.95 | 221.85 | 221.99 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 222.30 | 220.76 | 220.60 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 219.95 | 220.70 | 220.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 218.51 | 220.04 | 220.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 218.97 | 218.04 | 218.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 218.97 | 218.04 | 218.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 218.97 | 218.04 | 218.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 218.97 | 218.04 | 218.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 218.42 | 218.12 | 218.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 218.76 | 218.12 | 218.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 219.01 | 218.30 | 218.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 219.01 | 218.30 | 218.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 220.06 | 218.65 | 218.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 219.95 | 218.65 | 218.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 220.10 | 218.94 | 219.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 220.63 | 218.94 | 219.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 218.40 | 218.84 | 219.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 215.77 | 218.84 | 219.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 14:15:00 | 204.98 | 207.28 | 210.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 209.76 | 207.31 | 209.63 | SL hit (close>ema200) qty=0.50 sl=207.31 alert=retest2 |

### Cycle 186 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 213.36 | 210.29 | 210.19 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 209.20 | 211.20 | 211.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 208.65 | 210.69 | 211.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 207.27 | 207.24 | 208.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 207.41 | 207.24 | 208.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 209.38 | 207.66 | 208.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 209.38 | 207.66 | 208.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 206.96 | 207.52 | 208.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 205.78 | 207.51 | 208.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 205.57 | 206.96 | 207.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 205.81 | 206.56 | 207.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 208.49 | 207.45 | 207.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 208.49 | 207.45 | 207.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 211.00 | 209.06 | 208.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 214.98 | 215.53 | 213.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 214.98 | 215.53 | 213.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 214.63 | 215.43 | 214.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 214.63 | 215.43 | 214.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 215.46 | 215.43 | 214.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 215.78 | 215.50 | 214.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 216.18 | 215.71 | 215.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 214.22 | 215.29 | 215.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 214.22 | 215.29 | 215.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 213.95 | 215.02 | 215.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 206.00 | 204.51 | 206.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 206.00 | 204.51 | 206.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 207.73 | 205.16 | 206.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 207.73 | 205.16 | 206.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 204.90 | 205.10 | 206.44 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 207.61 | 206.76 | 206.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 208.25 | 207.33 | 207.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 209.57 | 209.66 | 209.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:30:00 | 210.21 | 209.66 | 209.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 211.28 | 210.33 | 209.61 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 207.54 | 209.55 | 209.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 206.75 | 208.99 | 209.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 208.00 | 207.87 | 208.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 208.00 | 207.87 | 208.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 208.00 | 207.87 | 208.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 208.00 | 207.87 | 208.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 208.31 | 207.96 | 208.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 208.31 | 207.96 | 208.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 208.32 | 208.03 | 208.56 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 211.32 | 209.28 | 209.03 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 207.80 | 209.27 | 209.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 206.73 | 208.31 | 208.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 205.74 | 205.32 | 206.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 205.74 | 205.32 | 206.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 206.65 | 205.76 | 206.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 206.40 | 205.76 | 206.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 209.31 | 206.47 | 206.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 209.31 | 206.47 | 206.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 211.65 | 207.51 | 207.36 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 207.31 | 208.62 | 208.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 205.93 | 208.08 | 208.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 207.73 | 207.30 | 207.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 207.73 | 207.30 | 207.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 207.73 | 207.30 | 207.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 207.80 | 207.30 | 207.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 207.60 | 207.36 | 207.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:30:00 | 207.91 | 207.36 | 207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 208.67 | 207.62 | 207.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 208.67 | 207.62 | 207.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 209.03 | 207.90 | 207.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 209.03 | 207.90 | 207.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 208.75 | 208.07 | 208.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 209.39 | 208.49 | 208.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 207.70 | 208.33 | 208.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 207.70 | 208.33 | 208.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 207.70 | 208.33 | 208.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 207.70 | 208.33 | 208.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 210.65 | 209.91 | 209.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 213.55 | 209.56 | 209.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 211.41 | 212.23 | 211.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 211.28 | 212.02 | 211.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 211.30 | 211.84 | 211.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 210.49 | 211.57 | 211.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 210.49 | 211.57 | 211.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 210.66 | 211.39 | 211.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 210.00 | 211.39 | 211.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 210.99 | 211.31 | 211.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 211.69 | 211.31 | 211.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 208.54 | 212.58 | 213.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 208.54 | 212.58 | 213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 207.23 | 211.51 | 212.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 188.96 | 188.62 | 192.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:45:00 | 189.33 | 188.62 | 192.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 185.30 | 184.46 | 185.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 184.35 | 184.46 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 184.67 | 184.50 | 185.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 183.80 | 184.50 | 185.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 184.05 | 184.41 | 185.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:00:00 | 183.99 | 184.34 | 185.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:30:00 | 184.03 | 184.32 | 185.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 182.55 | 183.93 | 184.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 182.22 | 183.93 | 184.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 182.44 | 183.49 | 184.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:30:00 | 182.52 | 182.86 | 183.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.61 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.85 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.79 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.83 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 173.11 | 175.47 | 177.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 173.32 | 175.47 | 177.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 173.39 | 175.47 | 177.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 173.41 | 173.01 | 174.58 | SL hit (close>ema200) qty=0.50 sl=173.01 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 175.80 | 175.13 | 175.05 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 174.70 | 175.01 | 175.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 172.93 | 174.42 | 174.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 171.65 | 171.63 | 172.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:30:00 | 171.68 | 171.63 | 172.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 172.56 | 171.81 | 172.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:45:00 | 172.65 | 171.81 | 172.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 172.40 | 171.93 | 172.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:15:00 | 172.47 | 171.93 | 172.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 172.53 | 172.05 | 172.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 172.43 | 172.05 | 172.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 172.24 | 172.09 | 172.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 171.15 | 172.14 | 172.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 162.59 | 165.25 | 167.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 163.41 | 163.21 | 164.98 | SL hit (close>ema200) qty=0.50 sl=163.21 alert=retest2 |

### Cycle 200 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 156.88 | 155.79 | 155.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 158.35 | 156.71 | 156.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 160.79 | 161.13 | 160.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 160.79 | 161.13 | 160.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 160.30 | 160.97 | 160.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 160.30 | 160.97 | 160.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 159.99 | 160.67 | 160.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 159.99 | 160.67 | 160.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 157.96 | 160.13 | 159.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 157.96 | 160.13 | 159.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 158.71 | 159.84 | 159.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 157.88 | 159.84 | 159.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 160.74 | 160.43 | 160.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 160.56 | 160.43 | 160.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 162.60 | 160.87 | 160.33 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 156.81 | 159.84 | 160.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 153.73 | 158.08 | 159.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 161.90 | 157.49 | 158.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 161.90 | 157.49 | 158.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 161.90 | 157.49 | 158.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 161.90 | 157.49 | 158.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 163.05 | 158.60 | 158.71 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 162.50 | 159.38 | 159.05 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 160.00 | 160.57 | 160.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 159.27 | 160.31 | 160.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 157.75 | 157.74 | 158.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 157.75 | 157.74 | 158.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 156.68 | 157.50 | 158.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 155.91 | 157.04 | 157.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 148.11 | 150.50 | 153.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 149.67 | 148.55 | 150.47 | SL hit (close>ema200) qty=0.50 sl=148.55 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 151.65 | 150.31 | 150.17 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 149.25 | 150.19 | 150.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 148.07 | 149.51 | 149.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 143.70 | 143.54 | 145.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 143.70 | 143.54 | 145.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 148.23 | 143.88 | 144.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 148.30 | 143.88 | 144.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 146.50 | 145.49 | 145.37 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 143.86 | 145.02 | 145.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 142.92 | 144.36 | 144.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 142.01 | 141.34 | 142.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 142.01 | 141.34 | 142.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 142.01 | 141.34 | 142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 144.31 | 141.34 | 142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 143.20 | 141.71 | 142.58 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 146.64 | 143.41 | 143.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 147.76 | 145.16 | 144.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 145.77 | 145.91 | 144.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 145.77 | 145.91 | 144.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 144.80 | 145.69 | 144.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 144.87 | 145.69 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 144.69 | 145.49 | 144.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 144.69 | 145.49 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 145.49 | 145.49 | 144.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 145.63 | 145.40 | 144.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 145.66 | 145.29 | 144.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 146.36 | 145.37 | 144.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:00:00 | 146.28 | 145.62 | 145.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 147.54 | 147.49 | 146.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 147.60 | 147.49 | 146.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 146.48 | 147.29 | 146.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 146.48 | 147.29 | 146.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 145.84 | 147.00 | 146.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 145.84 | 147.00 | 146.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 146.01 | 146.80 | 146.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 145.39 | 146.80 | 146.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 146.23 | 146.69 | 146.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 144.04 | 146.16 | 146.14 | SL hit (close<static) qty=1.00 sl=144.55 alert=retest2 |

### Cycle 209 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 144.62 | 145.85 | 146.00 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 147.94 | 146.19 | 146.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 150.98 | 147.44 | 146.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 153.59 | 154.46 | 152.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 153.59 | 154.46 | 152.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 153.59 | 154.46 | 152.47 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 146.37 | 151.57 | 151.86 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 156.80 | 152.46 | 152.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 157.95 | 154.65 | 153.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 157.94 | 158.61 | 156.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 157.94 | 158.61 | 156.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 157.90 | 158.47 | 157.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 155.94 | 158.47 | 157.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 155.87 | 157.95 | 156.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 154.77 | 157.95 | 156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 155.84 | 157.53 | 156.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 155.84 | 157.53 | 156.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 156.05 | 156.45 | 156.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 154.46 | 155.92 | 156.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 155.22 | 155.07 | 155.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 155.22 | 155.07 | 155.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 152.19 | 150.89 | 152.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 152.19 | 150.89 | 152.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 151.37 | 150.99 | 151.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:00:00 | 151.08 | 151.55 | 151.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-19 09:15:00 | 135.97 | 149.74 | 150.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 150.18 | 149.55 | 149.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 13:15:00 | 151.46 | 150.16 | 149.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 152.29 | 153.09 | 152.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 152.29 | 153.09 | 152.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 152.29 | 153.09 | 152.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 152.29 | 153.09 | 152.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 152.55 | 152.98 | 152.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 147.19 | 152.98 | 152.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 149.14 | 152.21 | 152.03 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 149.36 | 151.64 | 151.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 148.32 | 150.98 | 151.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 150.29 | 149.94 | 150.72 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 145.40 | 149.94 | 150.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 144.85 | 143.67 | 145.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 145.98 | 143.67 | 145.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 138.13 | 141.39 | 143.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 142.59 | 140.79 | 141.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 142.59 | 140.79 | 141.83 | SL hit (close>ema200) qty=0.50 sl=140.79 alert=retest1 |

### Cycle 216 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 144.46 | 142.42 | 142.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 145.10 | 143.63 | 143.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 146.56 | 147.90 | 146.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 149.38 | 148.19 | 146.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 149.38 | 148.19 | 146.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 149.85 | 148.52 | 146.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:30:00 | 150.71 | 149.02 | 147.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 149.67 | 149.10 | 147.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 144.10 | 147.38 | 147.27 | SL hit (close<static) qty=1.00 sl=146.50 alert=retest2 |

### Cycle 217 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 145.19 | 146.94 | 147.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 141.90 | 145.18 | 146.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 142.90 | 142.33 | 143.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:15:00 | 142.42 | 142.33 | 143.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 140.67 | 142.00 | 143.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 140.26 | 142.00 | 143.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 143.78 | 142.18 | 143.20 | SL hit (close>static) qty=1.00 sl=143.54 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 146.37 | 143.86 | 143.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 147.04 | 145.24 | 144.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 142.70 | 144.73 | 144.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 142.70 | 144.73 | 144.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 142.70 | 144.73 | 144.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 142.79 | 144.73 | 144.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 142.00 | 144.19 | 144.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 142.00 | 144.19 | 144.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 141.98 | 143.75 | 143.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 141.40 | 143.28 | 143.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 135.54 | 134.64 | 136.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 135.54 | 134.64 | 136.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 136.75 | 135.06 | 136.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 136.80 | 135.06 | 136.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 136.10 | 135.27 | 136.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 137.15 | 135.27 | 136.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 141.65 | 136.73 | 137.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 141.29 | 136.73 | 137.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 142.10 | 137.80 | 137.67 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 134.90 | 137.86 | 138.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 134.66 | 137.22 | 137.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 137.20 | 136.87 | 137.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 137.20 | 136.87 | 137.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 136.49 | 136.80 | 137.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 136.49 | 136.80 | 137.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 135.50 | 136.35 | 137.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:45:00 | 136.64 | 136.35 | 137.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 138.67 | 134.73 | 135.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 139.21 | 134.73 | 135.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 137.33 | 135.25 | 135.83 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 139.55 | 136.75 | 136.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 140.25 | 137.86 | 137.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 139.87 | 140.07 | 138.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 139.87 | 140.07 | 138.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 140.20 | 141.05 | 139.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 139.86 | 141.05 | 139.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 139.89 | 140.67 | 139.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:30:00 | 139.56 | 140.67 | 139.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 139.16 | 140.37 | 139.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 138.99 | 140.37 | 139.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 138.70 | 140.03 | 139.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 138.65 | 140.03 | 139.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 140.58 | 140.18 | 139.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 144.44 | 140.18 | 139.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 14:15:00 | 158.88 | 157.11 | 154.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 162.00 | 162.57 | 162.61 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 164.20 | 162.90 | 162.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 164.55 | 163.86 | 163.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 163.30 | 164.34 | 163.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 163.30 | 164.34 | 163.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 163.30 | 164.34 | 163.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 163.30 | 164.34 | 163.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 165.10 | 164.49 | 163.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:30:00 | 162.87 | 164.49 | 163.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 164.21 | 165.25 | 164.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 164.21 | 165.25 | 164.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 165.00 | 165.20 | 164.72 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 161.91 | 164.40 | 164.43 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 165.96 | 164.32 | 164.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 168.70 | 167.36 | 166.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 11:15:00 | 167.37 | 167.37 | 166.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:45:00 | 166.99 | 167.37 | 166.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 169.50 | 169.90 | 169.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 169.67 | 169.90 | 169.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 170.00 | 169.92 | 169.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 169.56 | 169.92 | 169.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 263.40 | 2024-04-12 10:15:00 | 260.45 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-04-12 09:45:00 | 262.70 | 2024-04-12 10:15:00 | 260.45 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-05-06 09:45:00 | 242.80 | 2024-05-06 11:15:00 | 247.20 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-05-08 13:30:00 | 245.50 | 2024-05-10 09:15:00 | 233.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 14:30:00 | 245.65 | 2024-05-10 09:15:00 | 233.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 15:15:00 | 244.25 | 2024-05-10 09:15:00 | 232.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 10:15:00 | 244.05 | 2024-05-10 09:15:00 | 231.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 13:30:00 | 245.50 | 2024-05-10 10:15:00 | 240.00 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2024-05-08 14:30:00 | 245.65 | 2024-05-10 10:15:00 | 240.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2024-05-08 15:15:00 | 244.25 | 2024-05-10 10:15:00 | 240.00 | STOP_HIT | 0.50 | 1.74% |
| SELL | retest2 | 2024-05-09 10:15:00 | 244.05 | 2024-05-10 10:15:00 | 240.00 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2024-05-09 14:15:00 | 238.60 | 2024-05-14 09:15:00 | 248.35 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2024-05-10 10:45:00 | 238.85 | 2024-05-14 09:15:00 | 248.35 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2024-05-13 09:15:00 | 238.95 | 2024-05-14 09:15:00 | 248.35 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2024-05-21 11:45:00 | 287.80 | 2024-05-24 15:15:00 | 281.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-05-21 15:00:00 | 288.05 | 2024-05-24 15:15:00 | 281.90 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-05-22 09:45:00 | 288.30 | 2024-05-24 15:15:00 | 281.90 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-05-22 12:45:00 | 287.90 | 2024-05-24 15:15:00 | 281.90 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-06-03 09:15:00 | 297.60 | 2024-06-04 10:15:00 | 271.60 | STOP_HIT | 1.00 | -8.74% |
| BUY | retest2 | 2024-06-12 10:30:00 | 333.00 | 2024-06-18 09:15:00 | 323.10 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-06-13 13:45:00 | 329.25 | 2024-06-18 09:15:00 | 323.10 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-06-14 11:15:00 | 332.00 | 2024-06-18 09:15:00 | 323.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-06-14 12:15:00 | 329.50 | 2024-06-18 09:15:00 | 323.10 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-06-25 09:30:00 | 333.90 | 2024-06-26 11:15:00 | 324.35 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-06-25 10:15:00 | 331.60 | 2024-06-26 11:15:00 | 324.35 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-06-25 12:00:00 | 331.70 | 2024-06-26 11:15:00 | 324.35 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-07-09 09:15:00 | 341.10 | 2024-07-09 10:15:00 | 334.95 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-11 10:15:00 | 327.65 | 2024-07-11 13:15:00 | 333.55 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-11 12:00:00 | 327.50 | 2024-07-11 13:15:00 | 333.55 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-07-12 15:00:00 | 327.75 | 2024-07-15 11:15:00 | 333.70 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-07-31 09:30:00 | 343.75 | 2024-08-02 11:15:00 | 343.85 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-08-02 10:00:00 | 342.70 | 2024-08-02 11:15:00 | 343.85 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-08-07 13:15:00 | 322.55 | 2024-08-09 15:15:00 | 324.65 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-08-07 14:30:00 | 323.00 | 2024-08-09 15:15:00 | 324.65 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-09 11:00:00 | 324.50 | 2024-08-09 15:15:00 | 324.65 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-08-09 12:00:00 | 324.20 | 2024-08-09 15:15:00 | 324.65 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-08-22 10:30:00 | 321.00 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-08-22 11:45:00 | 323.40 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-08-22 14:30:00 | 321.50 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-08-22 15:00:00 | 321.25 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-08-23 10:30:00 | 324.75 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-08-26 14:15:00 | 324.80 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-08-27 10:00:00 | 325.80 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-08-27 11:15:00 | 324.65 | 2024-08-29 10:15:00 | 321.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-09-18 12:15:00 | 313.10 | 2024-09-20 14:15:00 | 315.85 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-09-18 12:45:00 | 312.75 | 2024-09-20 14:15:00 | 315.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-09-19 09:45:00 | 311.40 | 2024-09-20 14:15:00 | 315.85 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-09-27 15:00:00 | 302.35 | 2024-10-03 10:15:00 | 307.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-30 13:00:00 | 303.40 | 2024-10-03 10:15:00 | 307.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-09-30 14:00:00 | 304.10 | 2024-10-03 10:15:00 | 307.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-10-01 12:00:00 | 302.75 | 2024-10-03 10:15:00 | 307.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-10-07 10:45:00 | 293.45 | 2024-10-09 09:15:00 | 302.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-10-07 11:45:00 | 295.65 | 2024-10-09 09:15:00 | 302.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-08 09:30:00 | 293.85 | 2024-10-09 09:15:00 | 302.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-10-08 10:00:00 | 295.60 | 2024-10-09 09:15:00 | 302.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-10-14 12:15:00 | 296.90 | 2024-10-15 11:15:00 | 301.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-10-18 11:30:00 | 311.45 | 2024-10-21 14:15:00 | 302.35 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-10-18 12:30:00 | 310.95 | 2024-10-21 14:15:00 | 302.35 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-10-18 13:00:00 | 311.05 | 2024-10-21 14:15:00 | 302.35 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest1 | 2024-10-31 09:15:00 | 297.55 | 2024-11-01 17:15:00 | 312.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-31 09:15:00 | 297.55 | 2024-11-04 15:15:00 | 307.65 | STOP_HIT | 0.50 | 3.39% |
| BUY | retest2 | 2024-11-05 15:00:00 | 308.05 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-11-06 09:15:00 | 310.10 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-11-06 10:00:00 | 307.50 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-11-06 12:00:00 | 307.65 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-07 15:00:00 | 310.10 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-11-08 10:30:00 | 309.80 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-11-08 11:00:00 | 309.85 | 2024-11-11 09:15:00 | 303.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-11-21 09:15:00 | 276.55 | 2024-11-22 13:15:00 | 284.15 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-11-21 12:00:00 | 276.10 | 2024-11-22 13:15:00 | 284.15 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-12-09 11:30:00 | 315.30 | 2024-12-10 09:15:00 | 309.35 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-12-09 12:15:00 | 315.25 | 2024-12-10 09:15:00 | 309.35 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-12-09 13:00:00 | 314.80 | 2024-12-10 09:15:00 | 309.35 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-12-09 13:45:00 | 314.85 | 2024-12-10 09:15:00 | 309.35 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-12-16 10:45:00 | 301.60 | 2024-12-16 13:15:00 | 308.20 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-12-27 10:15:00 | 284.00 | 2024-12-30 15:15:00 | 269.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 10:45:00 | 283.30 | 2024-12-30 15:15:00 | 269.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 10:15:00 | 284.00 | 2024-12-31 10:15:00 | 273.50 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2024-12-27 10:45:00 | 283.30 | 2024-12-31 10:15:00 | 273.50 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-01-15 13:30:00 | 241.00 | 2025-01-16 15:15:00 | 247.75 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-02-25 13:00:00 | 182.65 | 2025-02-28 12:15:00 | 173.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 13:00:00 | 182.65 | 2025-03-03 12:15:00 | 177.59 | STOP_HIT | 0.50 | 2.77% |
| BUY | retest2 | 2025-03-07 14:45:00 | 185.92 | 2025-03-10 14:15:00 | 183.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-03-10 10:45:00 | 186.44 | 2025-03-10 14:15:00 | 183.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-03-10 12:30:00 | 186.07 | 2025-03-10 14:15:00 | 183.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-03-11 12:15:00 | 182.77 | 2025-03-11 15:15:00 | 185.37 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-03-13 10:00:00 | 186.82 | 2025-03-17 14:15:00 | 184.35 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-03-17 09:15:00 | 187.83 | 2025-03-17 14:15:00 | 184.35 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-03-26 09:15:00 | 214.17 | 2025-04-04 10:15:00 | 208.90 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-04-09 09:15:00 | 203.82 | 2025-04-09 12:15:00 | 206.71 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-04-21 10:30:00 | 220.24 | 2025-04-23 09:15:00 | 216.30 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-04-21 11:15:00 | 220.48 | 2025-04-23 09:15:00 | 216.30 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-05-02 12:00:00 | 213.76 | 2025-05-05 09:15:00 | 217.64 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-05-02 13:30:00 | 213.72 | 2025-05-05 09:15:00 | 217.64 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-16 10:45:00 | 229.63 | 2025-05-23 09:15:00 | 231.51 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-05-19 09:15:00 | 233.20 | 2025-05-23 09:15:00 | 231.51 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-05-28 09:15:00 | 235.71 | 2025-05-30 09:15:00 | 233.15 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-06-20 12:15:00 | 220.60 | 2025-06-24 09:15:00 | 225.95 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-06-23 09:15:00 | 220.62 | 2025-06-24 09:15:00 | 225.95 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest1 | 2025-06-25 09:15:00 | 227.20 | 2025-07-02 09:15:00 | 229.00 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-06-30 09:30:00 | 231.21 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-07-01 10:00:00 | 231.15 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-07-01 11:30:00 | 231.20 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-07-01 12:00:00 | 231.40 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-14 13:45:00 | 222.82 | 2025-07-14 14:15:00 | 225.15 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-28 12:00:00 | 219.44 | 2025-07-29 14:15:00 | 222.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-08-06 11:30:00 | 214.49 | 2025-08-06 15:15:00 | 214.77 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-08-06 12:30:00 | 215.70 | 2025-08-06 15:15:00 | 214.77 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-08-13 14:30:00 | 223.03 | 2025-08-14 09:15:00 | 218.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-08-26 09:15:00 | 215.77 | 2025-08-29 14:15:00 | 204.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 215.77 | 2025-09-01 09:15:00 | 209.76 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2025-09-09 10:15:00 | 205.78 | 2025-09-11 09:15:00 | 208.49 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-09 12:00:00 | 205.57 | 2025-09-11 09:15:00 | 208.49 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-09 14:30:00 | 205.81 | 2025-09-11 09:15:00 | 208.49 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-18 15:00:00 | 215.78 | 2025-09-22 13:15:00 | 214.22 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-19 10:00:00 | 216.18 | 2025-09-22 13:15:00 | 214.22 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-27 09:15:00 | 213.55 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-10-28 11:15:00 | 211.41 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-28 11:45:00 | 211.28 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-28 12:30:00 | 211.30 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-29 09:15:00 | 211.69 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-17 11:15:00 | 183.80 | 2025-11-24 09:15:00 | 174.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 12:00:00 | 184.05 | 2025-11-24 09:15:00 | 174.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 14:00:00 | 183.99 | 2025-11-24 09:15:00 | 174.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 14:30:00 | 184.03 | 2025-11-24 09:15:00 | 174.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 10:15:00 | 182.22 | 2025-11-24 11:15:00 | 173.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 12:15:00 | 182.44 | 2025-11-24 11:15:00 | 173.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 14:30:00 | 182.52 | 2025-11-24 11:15:00 | 173.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:15:00 | 183.80 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2025-11-17 12:00:00 | 184.05 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2025-11-17 14:00:00 | 183.99 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.75% |
| SELL | retest2 | 2025-11-17 14:30:00 | 184.03 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-11-18 10:15:00 | 182.22 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-11-18 12:15:00 | 182.44 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-11-18 14:30:00 | 182.52 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-12-03 09:15:00 | 171.15 | 2025-12-08 13:15:00 | 162.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 171.15 | 2025-12-09 13:15:00 | 163.41 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2026-01-08 10:45:00 | 155.91 | 2026-01-09 15:15:00 | 148.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 155.91 | 2026-01-12 15:15:00 | 149.67 | STOP_HIT | 0.50 | 4.00% |
| BUY | retest2 | 2026-01-29 13:30:00 | 145.63 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-30 10:00:00 | 145.66 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-01-30 10:30:00 | 146.36 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-30 15:00:00 | 146.28 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-02-18 10:00:00 | 151.08 | 2026-02-19 09:15:00 | 135.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-03-04 09:15:00 | 145.40 | 2026-03-09 09:15:00 | 138.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-04 09:15:00 | 145.40 | 2026-03-10 09:15:00 | 142.59 | STOP_HIT | 0.50 | 1.93% |
| BUY | retest2 | 2026-03-12 12:00:00 | 149.85 | 2026-03-13 12:15:00 | 144.10 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-12 13:30:00 | 150.71 | 2026-03-13 12:15:00 | 144.10 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2026-03-12 14:30:00 | 149.67 | 2026-03-13 12:15:00 | 144.10 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-03-17 11:15:00 | 140.26 | 2026-03-17 13:15:00 | 143.78 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-08 09:15:00 | 144.44 | 2026-04-16 14:15:00 | 158.88 | TARGET_HIT | 1.00 | 10.00% |
