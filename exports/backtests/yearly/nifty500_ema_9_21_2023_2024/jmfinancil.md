# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 145.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 214 |
| ALERT1 | 151 |
| ALERT2 | 151 |
| ALERT2_SKIP | 103 |
| ALERT3 | 315 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 110 |
| PARTIAL | 35 |
| TARGET_HIT | 8 |
| STOP_HIT | 101 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 81 / 63
- **Target hits / Stop hits / Partials:** 8 / 101 / 35
- **Avg / median % per leg:** 1.62% / 2.53%
- **Sum % (uncompounded):** 233.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 10 | 24.4% | 3 | 38 | 0 | -0.27% | -11.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 41 | 10 | 24.4% | 3 | 38 | 0 | -0.27% | -11.0% |
| SELL (all) | 103 | 71 | 68.9% | 5 | 63 | 35 | 2.37% | 244.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 103 | 71 | 68.9% | 5 | 63 | 35 | 2.37% | 244.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 144 | 81 | 56.2% | 8 | 101 | 35 | 1.62% | 233.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 13:15:00 | 64.60 | 64.90 | 64.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 64.25 | 64.77 | 64.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 64.70 | 64.68 | 64.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 09:15:00 | 64.70 | 64.68 | 64.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 64.70 | 64.68 | 64.80 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 66.10 | 64.74 | 64.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 10:15:00 | 66.65 | 65.55 | 65.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 10:15:00 | 67.25 | 67.28 | 66.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 15:15:00 | 66.60 | 67.09 | 66.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 15:15:00 | 66.60 | 67.09 | 66.68 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 14:15:00 | 66.10 | 66.46 | 66.50 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 67.70 | 66.65 | 66.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 10:15:00 | 67.75 | 66.87 | 66.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 66.75 | 66.85 | 66.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 12:15:00 | 66.65 | 66.81 | 66.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 66.65 | 66.81 | 66.69 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 10:15:00 | 68.70 | 69.18 | 69.20 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 11:15:00 | 69.50 | 69.25 | 69.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 12:15:00 | 70.15 | 69.43 | 69.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 13:15:00 | 70.20 | 70.26 | 69.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 14:15:00 | 69.95 | 70.19 | 69.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 69.95 | 70.19 | 69.90 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 09:15:00 | 72.40 | 73.03 | 73.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 11:15:00 | 71.80 | 72.65 | 72.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 13:15:00 | 72.95 | 72.65 | 72.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 13:15:00 | 72.95 | 72.65 | 72.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 72.95 | 72.65 | 72.82 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 73.00 | 72.88 | 72.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 12:15:00 | 73.50 | 73.01 | 72.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 13:15:00 | 72.70 | 72.95 | 72.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 13:15:00 | 72.70 | 72.95 | 72.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 72.70 | 72.95 | 72.91 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 15:15:00 | 72.60 | 72.88 | 72.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 09:15:00 | 72.35 | 72.77 | 72.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 09:15:00 | 72.70 | 72.42 | 72.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 72.70 | 72.42 | 72.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 72.70 | 72.42 | 72.58 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 12:15:00 | 72.90 | 72.68 | 72.67 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 13:15:00 | 72.25 | 72.59 | 72.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 14:15:00 | 71.35 | 72.34 | 72.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 74.00 | 72.53 | 72.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 74.00 | 72.53 | 72.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 74.00 | 72.53 | 72.57 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 73.25 | 72.68 | 72.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 74.20 | 73.42 | 73.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 10:15:00 | 74.60 | 74.71 | 74.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 11:15:00 | 74.50 | 74.67 | 74.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 74.50 | 74.67 | 74.13 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 12:15:00 | 73.70 | 74.02 | 74.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 10:15:00 | 73.25 | 73.83 | 73.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 72.30 | 71.85 | 72.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 72.30 | 71.85 | 72.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 72.30 | 71.85 | 72.30 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 72.90 | 72.43 | 72.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 10:15:00 | 74.00 | 73.20 | 72.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-05 10:15:00 | 74.00 | 74.08 | 73.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 12:15:00 | 74.05 | 74.05 | 73.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 74.05 | 74.05 | 73.83 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 73.75 | 74.03 | 74.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 15:15:00 | 73.30 | 73.84 | 73.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 73.95 | 73.87 | 73.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 73.95 | 73.87 | 73.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 73.95 | 73.87 | 73.95 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 74.50 | 74.08 | 74.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 75.10 | 74.30 | 74.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 74.50 | 74.52 | 74.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 75.00 | 75.37 | 75.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 75.00 | 75.37 | 75.05 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 74.75 | 75.10 | 75.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 10:15:00 | 74.55 | 74.90 | 75.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 13:15:00 | 75.00 | 74.82 | 74.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 13:15:00 | 75.00 | 74.82 | 74.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 75.00 | 74.82 | 74.93 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 14:15:00 | 74.20 | 73.24 | 73.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 09:15:00 | 74.70 | 73.67 | 73.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 14:15:00 | 73.95 | 74.02 | 73.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 14:15:00 | 73.95 | 74.02 | 73.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 73.95 | 74.02 | 73.71 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 10:15:00 | 76.80 | 79.40 | 79.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 12:15:00 | 76.55 | 78.42 | 79.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 74.20 | 74.10 | 75.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 12:15:00 | 74.75 | 74.29 | 74.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 74.75 | 74.29 | 74.96 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 76.30 | 75.17 | 75.15 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 74.55 | 75.39 | 75.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 12:15:00 | 73.60 | 74.88 | 75.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 77.35 | 73.91 | 74.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 77.35 | 73.91 | 74.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 77.35 | 73.91 | 74.16 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 80.30 | 75.18 | 74.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 09:15:00 | 82.75 | 78.50 | 76.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 13:15:00 | 81.50 | 81.54 | 80.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 15:15:00 | 80.70 | 81.36 | 80.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 80.70 | 81.36 | 80.29 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 11:15:00 | 79.10 | 80.11 | 80.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 12:15:00 | 78.85 | 79.86 | 80.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 10:15:00 | 79.40 | 79.10 | 79.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 10:15:00 | 79.40 | 79.10 | 79.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 79.40 | 79.10 | 79.57 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 15:15:00 | 78.75 | 77.53 | 77.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 79.60 | 77.95 | 77.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 13:15:00 | 93.50 | 93.58 | 91.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 92.00 | 93.33 | 91.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 92.00 | 93.33 | 91.93 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 12:15:00 | 90.60 | 91.80 | 91.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 15:15:00 | 90.05 | 91.07 | 91.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 87.60 | 86.35 | 87.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 87.60 | 86.35 | 87.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 87.60 | 86.35 | 87.93 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 11:15:00 | 87.90 | 86.26 | 86.20 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 15:15:00 | 85.80 | 86.20 | 86.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 84.85 | 85.93 | 86.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 15:15:00 | 85.45 | 85.41 | 85.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 85.00 | 85.33 | 85.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 85.00 | 85.33 | 85.63 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 86.50 | 85.33 | 85.30 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 85.15 | 85.46 | 85.48 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 86.25 | 85.58 | 85.52 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 12:15:00 | 84.30 | 85.33 | 85.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 13:15:00 | 84.05 | 85.08 | 85.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 09:15:00 | 85.10 | 84.82 | 85.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 85.10 | 84.82 | 85.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 85.10 | 84.82 | 85.12 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 85.70 | 84.90 | 84.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 15:15:00 | 86.95 | 85.31 | 85.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 86.90 | 87.74 | 86.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 86.90 | 87.74 | 86.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 86.90 | 87.74 | 86.81 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 15:15:00 | 85.80 | 86.44 | 86.48 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 87.05 | 86.59 | 86.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 87.55 | 86.78 | 86.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 87.05 | 87.15 | 86.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 14:15:00 | 87.40 | 87.20 | 86.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 87.40 | 87.20 | 86.94 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 87.80 | 90.00 | 90.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 14:15:00 | 87.35 | 89.18 | 89.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 12:15:00 | 81.00 | 80.02 | 81.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 15:15:00 | 81.35 | 80.63 | 81.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 81.35 | 80.63 | 81.71 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 83.80 | 82.40 | 82.26 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 11:15:00 | 82.15 | 82.57 | 82.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 81.70 | 82.26 | 82.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 09:15:00 | 82.85 | 82.34 | 82.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 82.85 | 82.34 | 82.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 82.85 | 82.34 | 82.43 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 10:15:00 | 83.20 | 82.52 | 82.50 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 81.80 | 82.37 | 82.45 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 82.90 | 82.56 | 82.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 86.15 | 83.36 | 82.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 86.25 | 87.21 | 85.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 12:15:00 | 85.90 | 86.71 | 85.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 85.90 | 86.71 | 85.73 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 10:15:00 | 83.90 | 85.23 | 85.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 11:15:00 | 83.45 | 84.87 | 85.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 84.25 | 84.15 | 84.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 84.25 | 84.15 | 84.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 84.25 | 84.15 | 84.61 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 82.80 | 82.61 | 82.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 85.90 | 83.38 | 82.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 09:15:00 | 84.60 | 84.91 | 84.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 84.60 | 84.91 | 84.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 84.60 | 84.91 | 84.16 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 83.40 | 84.06 | 84.07 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 84.25 | 84.05 | 84.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 10:15:00 | 84.45 | 84.14 | 84.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 11:15:00 | 83.75 | 84.06 | 84.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 11:15:00 | 83.75 | 84.06 | 84.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 83.75 | 84.06 | 84.05 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 83.30 | 83.91 | 83.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 82.65 | 83.66 | 83.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 83.85 | 83.53 | 83.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 83.85 | 83.53 | 83.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 83.85 | 83.53 | 83.73 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 84.00 | 83.66 | 83.66 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 83.35 | 83.60 | 83.63 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 14:15:00 | 84.25 | 83.69 | 83.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 86.30 | 84.30 | 83.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 84.90 | 84.96 | 84.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 84.05 | 84.73 | 84.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 84.05 | 84.73 | 84.43 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 13:15:00 | 83.85 | 84.30 | 84.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 15:15:00 | 83.70 | 84.12 | 84.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 10:15:00 | 84.35 | 84.16 | 84.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 10:15:00 | 84.35 | 84.16 | 84.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 84.35 | 84.16 | 84.22 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 11:15:00 | 85.25 | 84.38 | 84.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 86.00 | 85.10 | 84.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 09:15:00 | 86.25 | 86.71 | 86.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 86.25 | 86.71 | 86.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 86.25 | 86.71 | 86.22 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 87.25 | 88.00 | 88.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 87.10 | 87.82 | 87.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 12:15:00 | 87.95 | 87.85 | 87.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 87.95 | 87.85 | 87.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 87.95 | 87.85 | 87.93 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 89.75 | 88.31 | 88.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 93.25 | 89.92 | 89.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 100.95 | 100.96 | 97.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 100.20 | 101.54 | 100.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 100.20 | 101.54 | 100.34 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 95.50 | 99.57 | 99.60 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 103.10 | 98.15 | 98.04 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 12:15:00 | 97.75 | 98.53 | 98.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 13:15:00 | 97.35 | 98.02 | 98.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 14:15:00 | 98.25 | 98.06 | 98.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 14:15:00 | 98.25 | 98.06 | 98.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 98.25 | 98.06 | 98.25 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 11:15:00 | 100.00 | 98.54 | 98.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 09:15:00 | 100.80 | 99.67 | 99.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 12:15:00 | 99.85 | 99.89 | 99.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 100.70 | 100.06 | 99.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 100.70 | 100.06 | 99.60 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 99.35 | 101.07 | 101.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 13:15:00 | 98.45 | 99.62 | 100.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 99.50 | 99.14 | 99.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 99.50 | 99.14 | 99.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 99.50 | 99.14 | 99.95 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 101.45 | 99.34 | 99.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 106.10 | 102.36 | 101.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 10:15:00 | 105.45 | 105.74 | 104.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 11:15:00 | 104.40 | 105.47 | 104.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 104.40 | 105.47 | 104.12 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 10:15:00 | 103.10 | 103.75 | 103.76 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 11:15:00 | 105.80 | 104.16 | 103.94 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 14:15:00 | 103.00 | 103.67 | 103.75 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 107.00 | 104.27 | 104.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 10:15:00 | 110.20 | 105.46 | 104.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 15:15:00 | 109.75 | 110.29 | 108.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 105.65 | 109.36 | 108.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 105.65 | 109.36 | 108.34 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 105.85 | 107.57 | 107.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 105.50 | 106.55 | 107.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 106.65 | 106.57 | 107.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 107.85 | 106.83 | 107.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 107.85 | 106.83 | 107.08 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 109.90 | 107.44 | 107.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 11:15:00 | 112.95 | 109.71 | 108.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 12:15:00 | 111.45 | 111.78 | 110.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 111.95 | 111.71 | 110.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 111.95 | 111.71 | 110.91 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 12:15:00 | 110.70 | 110.88 | 110.89 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 111.95 | 111.04 | 110.95 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 108.70 | 110.94 | 111.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 15:15:00 | 108.00 | 110.35 | 110.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 109.75 | 108.42 | 109.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 109.75 | 108.42 | 109.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 109.75 | 108.42 | 109.30 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 103.60 | 101.69 | 101.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 104.60 | 102.43 | 102.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 15:15:00 | 103.05 | 103.06 | 102.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 14:15:00 | 103.45 | 103.67 | 103.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 103.45 | 103.67 | 103.19 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 102.30 | 103.04 | 103.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 11:15:00 | 101.60 | 102.48 | 102.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 102.40 | 101.79 | 102.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 102.40 | 101.79 | 102.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 102.40 | 101.79 | 102.26 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 11:15:00 | 105.10 | 102.85 | 102.68 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 12:15:00 | 102.35 | 103.05 | 103.09 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 105.25 | 103.49 | 103.28 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 101.90 | 102.92 | 103.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 100.50 | 102.43 | 102.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 102.90 | 102.36 | 102.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 102.90 | 102.36 | 102.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 102.90 | 102.36 | 102.71 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 100.00 | 98.81 | 98.67 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 13:15:00 | 98.20 | 98.59 | 98.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 14:15:00 | 97.70 | 98.41 | 98.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 85.80 | 84.35 | 89.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 12:15:00 | 88.00 | 86.36 | 88.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 88.00 | 86.36 | 88.71 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 10:15:00 | 75.45 | 74.14 | 74.13 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 09:15:00 | 71.70 | 73.77 | 74.01 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 12:15:00 | 77.25 | 73.29 | 73.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 78.95 | 77.34 | 76.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 80.30 | 80.68 | 79.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 10:15:00 | 79.00 | 80.34 | 79.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 79.00 | 80.34 | 79.66 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 78.00 | 79.10 | 79.22 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 80.75 | 79.31 | 79.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 13:15:00 | 81.65 | 80.09 | 79.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 12:15:00 | 81.45 | 82.14 | 81.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 12:15:00 | 81.45 | 82.14 | 81.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 81.45 | 82.14 | 81.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 83.60 | 82.31 | 81.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 15:15:00 | 82.70 | 82.73 | 82.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 80.10 | 82.20 | 82.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 80.10 | 82.20 | 82.21 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 14:15:00 | 82.10 | 81.66 | 81.64 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 10:15:00 | 81.15 | 81.53 | 81.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 79.75 | 80.95 | 81.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 80.20 | 80.01 | 80.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 13:15:00 | 80.20 | 80.01 | 80.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 80.20 | 80.01 | 80.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 13:45:00 | 80.35 | 80.01 | 80.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 80.85 | 80.18 | 80.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 15:00:00 | 80.85 | 80.18 | 80.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 81.00 | 80.34 | 80.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:15:00 | 81.60 | 80.34 | 80.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 81.15 | 80.50 | 80.67 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 81.75 | 80.82 | 80.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 82.50 | 81.49 | 81.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 82.50 | 82.71 | 82.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:00:00 | 82.50 | 82.71 | 82.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 82.30 | 82.63 | 82.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 82.05 | 82.63 | 82.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 81.85 | 82.47 | 82.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 81.85 | 82.47 | 82.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 83.50 | 82.68 | 82.26 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 12:15:00 | 81.80 | 82.19 | 82.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 09:15:00 | 81.60 | 81.98 | 82.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 15:15:00 | 81.70 | 81.58 | 81.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-30 09:15:00 | 83.95 | 81.58 | 81.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 86 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 85.85 | 82.44 | 82.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 88.80 | 83.71 | 82.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 09:15:00 | 86.20 | 87.03 | 85.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 10:00:00 | 86.20 | 87.03 | 85.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 84.65 | 86.22 | 85.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:00:00 | 84.65 | 86.22 | 85.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 84.50 | 85.88 | 85.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 85.05 | 85.58 | 85.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 12:15:00 | 84.10 | 84.83 | 84.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 84.10 | 84.83 | 84.91 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 14:15:00 | 85.35 | 84.95 | 84.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 15:15:00 | 85.70 | 85.10 | 85.02 | Break + close above crossover candle high |

### Cycle 89 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 83.10 | 84.70 | 84.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 81.50 | 83.10 | 83.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 82.60 | 82.24 | 82.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 82.60 | 82.24 | 82.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 80.80 | 80.08 | 80.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:00:00 | 80.80 | 80.08 | 80.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 80.25 | 80.11 | 80.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 80.60 | 80.11 | 80.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 80.50 | 80.19 | 80.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 79.35 | 80.19 | 80.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 13:15:00 | 79.25 | 79.59 | 80.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 15:15:00 | 80.40 | 80.15 | 80.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 80.40 | 80.15 | 80.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 80.85 | 80.29 | 80.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 11:15:00 | 81.95 | 82.00 | 81.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 12:00:00 | 81.95 | 82.00 | 81.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 81.65 | 81.93 | 81.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 81.65 | 81.93 | 81.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 81.65 | 81.87 | 81.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:30:00 | 81.70 | 81.87 | 81.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 81.55 | 81.81 | 81.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 81.55 | 81.81 | 81.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 81.75 | 81.80 | 81.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 81.95 | 81.80 | 81.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:45:00 | 82.00 | 81.84 | 81.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 10:15:00 | 81.20 | 81.63 | 81.57 | SL hit (close<static) qty=1.00 sl=81.30 alert=retest2 |

### Cycle 91 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 79.75 | 81.54 | 81.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 78.90 | 79.94 | 80.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 79.85 | 79.18 | 79.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 79.85 | 79.18 | 79.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 79.85 | 79.18 | 79.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:45:00 | 80.20 | 79.18 | 79.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 80.20 | 79.38 | 79.85 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 82.30 | 80.40 | 80.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 84.40 | 81.42 | 80.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 14:15:00 | 81.80 | 82.21 | 81.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 15:00:00 | 81.80 | 82.21 | 81.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 80.95 | 81.92 | 81.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 80.95 | 81.92 | 81.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 81.10 | 81.76 | 81.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 81.55 | 81.52 | 81.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 81.50 | 81.36 | 81.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 80.25 | 81.16 | 81.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 80.25 | 81.16 | 81.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 79.90 | 80.60 | 80.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 80.00 | 79.48 | 80.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 80.00 | 79.48 | 80.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 80.00 | 79.48 | 80.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 76.85 | 79.59 | 79.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 73.01 | 76.94 | 78.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 69.16 | 76.45 | 78.11 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 94 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 79.15 | 77.23 | 77.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 80.80 | 78.99 | 78.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 79.51 | 79.53 | 78.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:45:00 | 79.43 | 79.53 | 78.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 78.96 | 79.35 | 78.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:00:00 | 78.96 | 79.35 | 78.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 78.49 | 79.18 | 78.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:00:00 | 78.49 | 79.18 | 78.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 77.97 | 78.94 | 78.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:45:00 | 78.00 | 78.94 | 78.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 09:15:00 | 78.00 | 78.63 | 78.64 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 80.43 | 78.94 | 78.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 85.03 | 80.78 | 79.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 11:15:00 | 84.04 | 84.11 | 82.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 12:00:00 | 84.04 | 84.11 | 82.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 82.79 | 83.97 | 83.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 83.22 | 83.97 | 83.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 83.05 | 83.78 | 83.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 82.65 | 83.78 | 83.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 82.29 | 83.22 | 83.34 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 84.43 | 83.50 | 83.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 85.60 | 83.92 | 83.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 83.68 | 85.34 | 84.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 83.68 | 85.34 | 84.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 83.68 | 85.34 | 84.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 83.68 | 85.34 | 84.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 83.45 | 84.96 | 84.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 83.11 | 84.96 | 84.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 83.74 | 84.60 | 84.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 83.74 | 84.60 | 84.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 83.17 | 84.20 | 84.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 82.85 | 83.83 | 84.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 82.96 | 82.89 | 83.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 82.96 | 82.89 | 83.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 82.96 | 82.89 | 83.20 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 83.93 | 83.42 | 83.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 84.66 | 83.67 | 83.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 96.00 | 96.10 | 94.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 12:15:00 | 94.93 | 95.74 | 94.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 94.93 | 95.74 | 94.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 95.19 | 95.74 | 94.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 94.63 | 95.52 | 94.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 94.63 | 95.52 | 94.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 95.01 | 95.42 | 94.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 96.31 | 95.10 | 94.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 92.78 | 94.64 | 94.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 92.78 | 94.64 | 94.64 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 13:15:00 | 94.50 | 93.67 | 93.66 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 10:15:00 | 93.00 | 93.59 | 93.64 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 94.84 | 93.84 | 93.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 100.35 | 95.73 | 94.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 98.40 | 98.86 | 97.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 98.40 | 98.86 | 97.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 97.50 | 98.63 | 97.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 97.50 | 98.63 | 97.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 97.38 | 98.38 | 97.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 97.38 | 98.38 | 97.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 97.25 | 98.15 | 97.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:15:00 | 97.65 | 98.15 | 97.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 97.65 | 98.05 | 97.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 96.59 | 98.05 | 97.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 94.90 | 97.42 | 97.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 94.90 | 97.42 | 97.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 94.70 | 96.88 | 97.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 94.40 | 95.76 | 96.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 95.05 | 94.99 | 95.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 94.82 | 94.99 | 95.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 95.80 | 94.71 | 95.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 96.00 | 94.71 | 95.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 96.50 | 95.07 | 95.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 97.22 | 95.07 | 95.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 97.11 | 95.48 | 95.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 99.10 | 96.20 | 95.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 102.15 | 102.47 | 100.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 102.11 | 102.47 | 100.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 104.07 | 104.37 | 103.74 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 99.85 | 102.96 | 103.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 10:15:00 | 97.89 | 98.79 | 100.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 14:15:00 | 98.32 | 98.19 | 99.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 15:00:00 | 98.32 | 98.19 | 99.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 98.97 | 98.22 | 99.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:15:00 | 97.52 | 98.23 | 98.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 92.64 | 94.42 | 95.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 94.35 | 91.84 | 92.64 | SL hit (close>ema200) qty=0.50 sl=91.84 alert=retest2 |

### Cycle 108 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 94.00 | 93.12 | 93.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 95.25 | 94.12 | 93.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 93.27 | 93.95 | 93.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 10:15:00 | 93.27 | 93.95 | 93.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 93.27 | 93.95 | 93.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 93.27 | 93.95 | 93.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 93.29 | 93.82 | 93.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:30:00 | 93.45 | 93.82 | 93.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 93.17 | 93.69 | 93.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:45:00 | 93.10 | 93.69 | 93.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 92.79 | 93.40 | 93.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 92.03 | 92.74 | 93.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 94.20 | 92.62 | 92.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 94.20 | 92.62 | 92.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 94.20 | 92.62 | 92.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 94.15 | 92.62 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 96.77 | 93.45 | 93.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 11:15:00 | 99.30 | 94.62 | 93.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 102.32 | 102.43 | 99.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:00:00 | 102.32 | 102.43 | 99.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 110.60 | 103.95 | 101.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:45:00 | 113.50 | 109.73 | 108.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-05 09:15:00 | 124.85 | 121.98 | 117.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 127.44 | 130.15 | 130.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 126.70 | 129.07 | 129.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 125.18 | 124.08 | 125.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 125.18 | 124.08 | 125.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 125.18 | 124.08 | 125.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 126.22 | 124.08 | 125.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 125.49 | 124.36 | 125.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 125.49 | 124.36 | 125.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 125.62 | 124.61 | 125.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 125.63 | 124.61 | 125.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 125.41 | 124.77 | 125.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:30:00 | 126.38 | 124.77 | 125.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 125.61 | 124.94 | 125.71 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 128.81 | 126.10 | 126.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 11:15:00 | 131.14 | 128.75 | 127.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 136.42 | 138.57 | 135.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:00:00 | 136.42 | 138.57 | 135.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 136.36 | 138.13 | 135.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:15:00 | 135.00 | 138.13 | 135.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 135.56 | 137.61 | 135.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:30:00 | 135.12 | 137.61 | 135.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 135.29 | 137.15 | 135.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 135.29 | 137.15 | 135.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 150.76 | 152.40 | 149.77 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 143.97 | 148.34 | 148.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 140.18 | 144.05 | 145.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 141.08 | 139.95 | 142.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 141.08 | 139.95 | 142.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 141.08 | 139.95 | 142.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 143.94 | 139.95 | 142.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 143.12 | 139.18 | 140.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 143.12 | 139.18 | 140.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 140.65 | 139.47 | 140.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 140.03 | 139.61 | 140.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 145.80 | 141.85 | 141.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 145.80 | 141.85 | 141.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 147.86 | 144.53 | 143.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 157.08 | 159.55 | 157.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 157.08 | 159.55 | 157.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 157.08 | 159.55 | 157.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 157.08 | 159.55 | 157.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 157.77 | 159.20 | 157.65 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 153.67 | 156.81 | 157.06 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 159.84 | 156.81 | 156.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 162.16 | 157.88 | 157.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 157.50 | 158.05 | 157.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 11:15:00 | 157.50 | 158.05 | 157.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 157.50 | 158.05 | 157.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 157.50 | 158.05 | 157.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 156.26 | 157.69 | 157.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 156.26 | 157.69 | 157.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 146.15 | 155.38 | 156.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 143.73 | 150.68 | 153.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 143.00 | 142.88 | 147.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:45:00 | 143.69 | 142.88 | 147.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 131.90 | 135.15 | 138.04 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 139.57 | 137.36 | 137.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 142.06 | 139.63 | 138.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 136.44 | 139.35 | 138.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 136.44 | 139.35 | 138.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 136.44 | 139.35 | 138.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 136.44 | 139.35 | 138.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 136.25 | 138.73 | 138.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 136.25 | 138.73 | 138.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 136.95 | 138.38 | 138.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 135.09 | 137.37 | 137.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 138.42 | 137.29 | 137.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 138.42 | 137.29 | 137.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 138.42 | 137.29 | 137.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 138.93 | 137.29 | 137.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 137.77 | 137.39 | 137.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 137.20 | 137.58 | 137.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 138.79 | 137.82 | 137.92 | SL hit (close>static) qty=1.00 sl=138.74 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 140.33 | 138.32 | 138.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 143.55 | 140.04 | 139.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 145.38 | 147.00 | 145.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 10:15:00 | 145.38 | 147.00 | 145.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 145.38 | 147.00 | 145.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 145.38 | 147.00 | 145.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 145.46 | 146.69 | 145.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:30:00 | 146.53 | 146.05 | 145.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 15:15:00 | 146.25 | 146.05 | 145.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 144.70 | 145.81 | 145.27 | SL hit (close<static) qty=1.00 sl=144.90 alert=retest2 |

### Cycle 121 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 143.15 | 144.66 | 144.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 139.91 | 143.71 | 144.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 132.62 | 132.31 | 135.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 132.62 | 132.31 | 135.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 129.35 | 129.56 | 131.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 133.32 | 129.56 | 131.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 134.24 | 130.49 | 131.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 134.24 | 130.49 | 131.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 133.50 | 131.09 | 131.67 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 133.42 | 132.02 | 132.02 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 131.50 | 132.04 | 132.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 130.68 | 131.77 | 131.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 12:15:00 | 131.75 | 131.72 | 131.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-21 13:00:00 | 131.75 | 131.72 | 131.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 131.51 | 131.68 | 131.82 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 133.12 | 132.03 | 131.95 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 132.49 | 133.14 | 133.20 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 133.83 | 133.15 | 133.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 137.10 | 133.94 | 133.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 135.10 | 135.80 | 134.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 135.10 | 135.80 | 134.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 134.90 | 135.62 | 134.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 135.89 | 135.45 | 134.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:15:00 | 137.88 | 135.57 | 135.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 14:15:00 | 139.96 | 141.34 | 141.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 139.96 | 141.34 | 141.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 137.89 | 139.73 | 140.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 139.30 | 138.68 | 139.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 139.30 | 138.68 | 139.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 139.30 | 138.68 | 139.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 139.30 | 138.68 | 139.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 139.90 | 138.92 | 139.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 140.86 | 138.92 | 139.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 140.31 | 139.20 | 139.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 140.59 | 139.20 | 139.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 139.66 | 139.29 | 139.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 140.25 | 139.29 | 139.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 139.55 | 139.34 | 139.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 139.86 | 139.34 | 139.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 140.18 | 139.51 | 139.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 140.18 | 139.51 | 139.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 139.61 | 139.53 | 139.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:45:00 | 139.03 | 139.48 | 139.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 132.08 | 135.20 | 136.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 133.70 | 133.58 | 135.09 | SL hit (close>ema200) qty=0.50 sl=133.58 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 130.00 | 127.36 | 127.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 131.47 | 129.71 | 129.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 130.47 | 132.62 | 131.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 130.47 | 132.62 | 131.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 130.47 | 132.62 | 131.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 130.47 | 132.62 | 131.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 129.80 | 132.05 | 131.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 126.85 | 132.05 | 131.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 126.69 | 130.98 | 131.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 123.63 | 129.51 | 130.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 124.18 | 124.06 | 125.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:30:00 | 123.72 | 124.06 | 125.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 123.82 | 122.28 | 123.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 123.82 | 122.28 | 123.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 122.56 | 122.34 | 123.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 122.34 | 122.36 | 123.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 122.18 | 122.36 | 123.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 122.40 | 122.59 | 123.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 116.22 | 119.02 | 120.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 116.07 | 119.02 | 120.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 116.28 | 119.02 | 120.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 115.49 | 115.20 | 117.58 | SL hit (close>ema200) qty=0.50 sl=115.20 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 117.72 | 117.52 | 117.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 120.34 | 118.09 | 117.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 118.24 | 119.06 | 118.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 118.24 | 119.06 | 118.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 118.24 | 119.06 | 118.59 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 117.16 | 118.11 | 118.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 116.71 | 117.83 | 118.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 118.41 | 117.68 | 117.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 118.41 | 117.68 | 117.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 118.41 | 117.68 | 117.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:30:00 | 118.08 | 117.68 | 117.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 119.38 | 118.02 | 118.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:00:00 | 119.38 | 118.02 | 118.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 120.29 | 118.48 | 118.24 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 117.39 | 118.39 | 118.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 116.97 | 118.11 | 118.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 115.04 | 113.97 | 115.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 115.04 | 113.97 | 115.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 115.04 | 113.97 | 115.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 115.20 | 113.97 | 115.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 115.62 | 114.30 | 115.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 115.84 | 114.30 | 115.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 115.10 | 114.46 | 115.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:30:00 | 114.47 | 114.64 | 115.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 114.39 | 114.64 | 115.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 108.75 | 110.78 | 112.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 108.67 | 110.78 | 112.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 107.82 | 107.31 | 109.32 | SL hit (close>ema200) qty=0.50 sl=107.31 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 109.20 | 107.27 | 107.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 110.62 | 107.94 | 107.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 110.77 | 111.20 | 109.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 110.77 | 111.20 | 109.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 108.01 | 111.00 | 110.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 107.40 | 111.00 | 110.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 108.13 | 110.11 | 109.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:45:00 | 108.26 | 110.11 | 109.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 108.24 | 109.73 | 109.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 106.44 | 108.38 | 109.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 13:15:00 | 107.65 | 107.51 | 108.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 13:15:00 | 107.65 | 107.51 | 108.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 107.65 | 107.51 | 108.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 107.65 | 107.51 | 108.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 107.92 | 107.59 | 108.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 107.92 | 107.59 | 108.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 108.57 | 107.81 | 108.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 108.76 | 107.81 | 108.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 109.10 | 108.06 | 108.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 109.50 | 108.06 | 108.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 109.70 | 108.57 | 108.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 12:15:00 | 110.61 | 109.35 | 108.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 111.81 | 114.56 | 112.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 111.81 | 114.56 | 112.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 111.81 | 114.56 | 112.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 111.81 | 114.56 | 112.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 110.07 | 113.66 | 112.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 110.07 | 113.66 | 112.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 108.90 | 111.38 | 111.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 108.55 | 110.43 | 111.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 103.80 | 102.66 | 104.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 103.80 | 102.66 | 104.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 103.80 | 102.66 | 104.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 103.70 | 102.66 | 104.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 103.52 | 102.83 | 104.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 103.91 | 102.83 | 104.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 98.65 | 98.67 | 100.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:45:00 | 99.45 | 98.67 | 100.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 98.27 | 98.59 | 100.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 100.50 | 98.59 | 100.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 99.54 | 98.73 | 100.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 99.54 | 98.73 | 100.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 100.00 | 98.99 | 100.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 100.90 | 98.99 | 100.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 100.42 | 99.27 | 100.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 100.42 | 99.27 | 100.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 100.50 | 99.52 | 100.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 98.41 | 99.52 | 100.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 100.82 | 98.27 | 98.88 | SL hit (close>static) qty=1.00 sl=100.72 alert=retest2 |

### Cycle 138 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 101.40 | 99.27 | 99.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 102.04 | 100.41 | 99.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 102.93 | 102.95 | 102.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:30:00 | 102.40 | 102.95 | 102.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 101.80 | 102.84 | 102.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 100.12 | 102.84 | 102.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 100.77 | 102.42 | 102.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:15:00 | 101.11 | 102.42 | 102.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 101.16 | 101.92 | 101.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 101.16 | 101.92 | 101.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 12:15:00 | 100.64 | 101.18 | 101.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 15:15:00 | 90.95 | 90.78 | 92.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:15:00 | 91.88 | 90.78 | 92.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 92.66 | 91.15 | 92.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 92.66 | 91.15 | 92.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 92.82 | 91.49 | 92.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 92.36 | 91.66 | 92.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 92.00 | 92.24 | 92.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 94.61 | 92.67 | 92.85 | SL hit (close>static) qty=1.00 sl=93.55 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 94.53 | 93.05 | 93.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 95.73 | 93.58 | 93.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 95.94 | 96.51 | 95.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:45:00 | 96.11 | 96.51 | 95.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 95.48 | 96.31 | 95.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 97.25 | 96.31 | 95.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 93.39 | 96.46 | 96.20 | SL hit (close<static) qty=1.00 sl=95.22 alert=retest2 |

### Cycle 141 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 91.72 | 95.51 | 95.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 91.10 | 93.98 | 95.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 91.86 | 90.82 | 92.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 91.86 | 90.82 | 92.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 91.86 | 90.82 | 92.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:45:00 | 91.90 | 90.82 | 92.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 89.78 | 90.54 | 91.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 91.05 | 90.54 | 91.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 90.86 | 89.23 | 90.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:45:00 | 90.96 | 89.23 | 90.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 89.72 | 89.33 | 90.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 88.85 | 89.28 | 90.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:30:00 | 88.80 | 88.83 | 89.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 13:15:00 | 84.41 | 86.87 | 88.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 13:15:00 | 84.36 | 86.87 | 88.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 87.86 | 86.24 | 87.63 | SL hit (close>ema200) qty=0.50 sl=86.24 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 91.10 | 88.00 | 87.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 91.38 | 88.68 | 88.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 91.29 | 91.80 | 90.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 91.29 | 91.80 | 90.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 95.98 | 97.67 | 96.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 95.98 | 97.67 | 96.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 95.52 | 97.24 | 96.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 95.52 | 97.24 | 96.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 95.12 | 96.82 | 96.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 94.72 | 96.82 | 96.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 96.22 | 96.56 | 96.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 99.60 | 96.12 | 96.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 96.64 | 96.75 | 96.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 95.30 | 96.46 | 96.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 95.30 | 96.46 | 96.51 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 96.99 | 96.53 | 96.48 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 95.74 | 96.34 | 96.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 95.10 | 96.14 | 96.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 96.85 | 96.28 | 96.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 96.85 | 96.28 | 96.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 96.85 | 96.28 | 96.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 96.85 | 96.28 | 96.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 95.64 | 96.15 | 96.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 95.12 | 96.11 | 96.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 15:15:00 | 96.60 | 96.33 | 96.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 96.60 | 96.33 | 96.33 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 93.85 | 95.84 | 96.10 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 97.64 | 96.35 | 96.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 98.95 | 97.41 | 96.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 95.86 | 97.95 | 97.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 95.86 | 97.95 | 97.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 95.86 | 97.95 | 97.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 95.86 | 97.95 | 97.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 96.43 | 97.65 | 97.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 96.09 | 97.65 | 97.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 95.50 | 96.94 | 96.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 94.90 | 96.53 | 96.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 90.50 | 89.46 | 91.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 90.50 | 89.46 | 91.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 90.50 | 89.46 | 91.94 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 94.49 | 91.90 | 91.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 95.09 | 93.32 | 92.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 103.80 | 104.81 | 103.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 103.80 | 104.81 | 103.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 104.00 | 104.65 | 103.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 103.36 | 104.65 | 103.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 104.49 | 105.04 | 104.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 104.50 | 105.04 | 104.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 104.16 | 104.87 | 104.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 104.16 | 104.87 | 104.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 104.30 | 104.75 | 104.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 104.04 | 104.75 | 104.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 102.45 | 104.29 | 104.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 102.45 | 104.29 | 104.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 101.01 | 103.64 | 103.91 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 104.39 | 103.57 | 103.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 104.88 | 104.01 | 103.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 10:15:00 | 103.92 | 103.99 | 103.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 10:15:00 | 103.92 | 103.99 | 103.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 103.92 | 103.99 | 103.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 103.92 | 103.99 | 103.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 103.34 | 103.86 | 103.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 103.34 | 103.86 | 103.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 102.61 | 103.61 | 103.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 102.61 | 103.61 | 103.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 102.70 | 103.43 | 103.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 102.17 | 102.94 | 103.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 101.19 | 100.86 | 101.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 15:00:00 | 101.19 | 100.86 | 101.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 101.55 | 101.00 | 101.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 101.82 | 101.00 | 101.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 101.61 | 101.12 | 101.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 101.77 | 101.12 | 101.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 101.58 | 101.21 | 101.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 101.18 | 101.25 | 101.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 15:15:00 | 101.20 | 101.24 | 101.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 101.59 | 100.39 | 100.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 101.59 | 100.39 | 100.35 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 98.20 | 100.12 | 100.29 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 104.45 | 100.51 | 100.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 106.10 | 103.92 | 102.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 117.65 | 117.82 | 115.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 117.65 | 117.82 | 115.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 116.00 | 117.29 | 116.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 116.00 | 117.29 | 116.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 116.65 | 117.16 | 116.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 118.30 | 117.40 | 116.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 130.13 | 126.71 | 124.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 146.03 | 148.45 | 148.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 145.99 | 147.67 | 148.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 142.87 | 142.69 | 144.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:45:00 | 142.88 | 142.69 | 144.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 144.59 | 143.18 | 144.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 144.59 | 143.18 | 144.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 143.00 | 143.14 | 143.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 142.42 | 142.95 | 143.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 141.83 | 143.33 | 143.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 142.15 | 143.24 | 143.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 142.05 | 141.30 | 141.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 141.40 | 141.41 | 141.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 141.90 | 141.41 | 141.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 140.17 | 141.16 | 141.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 139.74 | 140.55 | 141.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 139.61 | 140.41 | 140.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 152.88 | 142.58 | 141.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 152.88 | 142.58 | 141.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 156.21 | 150.03 | 146.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 154.59 | 154.61 | 151.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 154.59 | 154.61 | 151.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 158.99 | 160.07 | 158.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 158.68 | 160.07 | 158.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 158.85 | 159.71 | 158.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 158.64 | 159.71 | 158.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 158.96 | 159.77 | 158.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 159.24 | 159.77 | 158.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 159.32 | 159.68 | 158.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 159.32 | 159.68 | 158.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 158.50 | 159.44 | 158.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 159.50 | 159.44 | 158.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 159.71 | 159.25 | 158.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 159.40 | 159.27 | 158.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:30:00 | 161.58 | 159.95 | 159.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 163.22 | 161.02 | 159.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 159.46 | 161.02 | 159.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 165.27 | 164.82 | 162.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 163.40 | 164.82 | 162.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 167.07 | 165.86 | 164.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 173.23 | 166.26 | 165.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 169.36 | 167.56 | 166.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:30:00 | 169.90 | 167.98 | 166.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 169.06 | 167.98 | 166.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 167.40 | 169.30 | 168.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 166.80 | 169.30 | 168.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 166.73 | 168.79 | 168.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 166.73 | 168.79 | 168.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 169.04 | 168.93 | 168.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 168.70 | 168.93 | 168.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 168.10 | 168.79 | 168.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 170.20 | 168.79 | 168.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 167.98 | 168.63 | 168.50 | SL hit (close<static) qty=1.00 sl=168.10 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 166.25 | 168.15 | 168.30 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 171.10 | 168.11 | 168.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 171.65 | 169.35 | 168.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 169.90 | 170.47 | 169.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 169.90 | 170.47 | 169.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 169.90 | 170.47 | 169.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 170.01 | 170.47 | 169.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 171.65 | 170.71 | 169.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 172.21 | 170.71 | 169.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 169.75 | 170.29 | 170.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 169.75 | 170.29 | 170.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 12:15:00 | 169.62 | 170.15 | 170.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 13:15:00 | 169.38 | 168.96 | 169.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 169.38 | 168.96 | 169.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 169.38 | 168.96 | 169.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 169.43 | 168.96 | 169.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 169.23 | 169.02 | 169.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 169.55 | 169.02 | 169.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 169.00 | 169.01 | 169.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 166.50 | 169.01 | 169.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 168.10 | 168.29 | 168.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 170.69 | 168.82 | 168.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 170.69 | 168.82 | 168.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 176.46 | 170.78 | 169.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 176.66 | 176.84 | 174.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:00:00 | 176.66 | 176.84 | 174.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 172.38 | 175.97 | 174.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 172.38 | 175.97 | 174.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 169.89 | 174.75 | 174.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 169.89 | 174.75 | 174.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 169.31 | 173.67 | 173.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 168.78 | 171.32 | 172.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 163.40 | 163.11 | 166.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 163.40 | 163.11 | 166.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 156.48 | 155.10 | 156.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 156.50 | 155.10 | 156.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 156.74 | 155.43 | 156.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 156.74 | 155.43 | 156.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 157.18 | 155.78 | 156.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 157.18 | 155.78 | 156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 157.50 | 156.12 | 156.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 157.94 | 156.12 | 156.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 154.96 | 155.89 | 156.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 154.39 | 155.89 | 156.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 154.25 | 155.24 | 156.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:45:00 | 153.80 | 154.33 | 155.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:30:00 | 154.43 | 154.45 | 154.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 154.06 | 154.26 | 154.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 155.01 | 154.26 | 154.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 154.24 | 154.26 | 154.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 154.24 | 154.26 | 154.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 155.04 | 154.06 | 154.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 155.04 | 154.06 | 154.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 155.20 | 154.29 | 154.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 158.09 | 154.29 | 154.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 157.66 | 154.96 | 154.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 157.66 | 154.96 | 154.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 160.86 | 157.25 | 156.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 159.80 | 160.17 | 158.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 159.80 | 160.17 | 158.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 182.52 | 185.10 | 182.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 182.52 | 185.10 | 182.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 186.50 | 185.38 | 182.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:15:00 | 187.05 | 185.62 | 182.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 187.00 | 186.09 | 183.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 192.97 | 185.30 | 184.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 186.90 | 191.00 | 191.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 186.90 | 191.00 | 191.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 185.55 | 187.68 | 189.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 183.20 | 181.15 | 183.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 183.20 | 181.15 | 183.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 183.20 | 181.15 | 183.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 183.95 | 181.15 | 183.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 182.75 | 181.47 | 183.43 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 188.76 | 184.37 | 184.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 191.80 | 186.48 | 185.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 183.21 | 187.27 | 186.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 183.21 | 187.27 | 186.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 183.21 | 187.27 | 186.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 182.70 | 187.27 | 186.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 181.90 | 186.19 | 185.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 181.90 | 186.19 | 185.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 183.06 | 185.57 | 185.57 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 188.49 | 185.29 | 185.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 189.06 | 186.04 | 185.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 189.94 | 190.12 | 188.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 189.94 | 190.12 | 188.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 190.35 | 190.60 | 189.44 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 184.51 | 188.33 | 188.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 180.35 | 186.73 | 187.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 180.73 | 180.65 | 183.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:45:00 | 180.63 | 180.65 | 183.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 182.45 | 180.98 | 182.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:45:00 | 180.54 | 181.05 | 182.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 180.70 | 180.87 | 182.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 179.92 | 180.39 | 181.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 180.85 | 180.08 | 180.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 180.00 | 180.06 | 180.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 180.60 | 180.06 | 180.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 177.33 | 178.67 | 179.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 176.94 | 178.32 | 179.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 176.91 | 178.32 | 179.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 176.52 | 177.89 | 179.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 176.74 | 176.49 | 177.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 12:15:00 | 171.66 | 174.02 | 175.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 12:15:00 | 171.81 | 174.02 | 175.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 14:15:00 | 171.51 | 173.37 | 175.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 173.21 | 173.12 | 174.64 | SL hit (close>ema200) qty=0.50 sl=173.12 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 165.69 | 162.91 | 162.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 168.29 | 164.91 | 163.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 175.72 | 176.65 | 174.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 175.72 | 176.65 | 174.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 175.97 | 176.43 | 175.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 176.69 | 176.43 | 175.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 173.00 | 175.75 | 175.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 173.09 | 175.75 | 175.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 172.55 | 175.11 | 175.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:30:00 | 172.88 | 175.11 | 175.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 172.47 | 174.58 | 174.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 170.81 | 173.45 | 174.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 173.94 | 173.19 | 173.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 173.94 | 173.19 | 173.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 173.94 | 173.19 | 173.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 173.91 | 173.19 | 173.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 174.88 | 173.53 | 173.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 174.60 | 173.53 | 173.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 175.30 | 173.88 | 174.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 175.72 | 173.88 | 174.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 176.50 | 174.41 | 174.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 177.82 | 175.09 | 174.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 174.56 | 175.37 | 174.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 174.56 | 175.37 | 174.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 174.56 | 175.37 | 174.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 174.56 | 175.37 | 174.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 173.60 | 175.02 | 174.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 173.60 | 175.02 | 174.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 172.81 | 174.57 | 174.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 172.60 | 173.77 | 174.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 174.68 | 173.95 | 174.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 174.68 | 173.95 | 174.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 174.68 | 173.95 | 174.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 174.68 | 173.95 | 174.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 174.65 | 174.09 | 174.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 175.03 | 174.09 | 174.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 175.10 | 174.29 | 174.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 175.56 | 174.29 | 174.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 175.70 | 174.57 | 174.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 178.05 | 175.48 | 174.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 176.31 | 176.89 | 175.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 176.31 | 176.89 | 175.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 175.20 | 176.55 | 175.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 175.20 | 176.55 | 175.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 176.15 | 176.47 | 175.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 173.76 | 176.47 | 175.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 175.61 | 176.30 | 175.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 175.75 | 176.30 | 175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 174.75 | 175.99 | 175.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 173.98 | 175.99 | 175.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 171.35 | 175.06 | 175.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 169.70 | 171.20 | 171.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 170.55 | 169.76 | 170.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 170.55 | 169.76 | 170.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 170.55 | 169.76 | 170.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 170.55 | 169.76 | 170.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 171.20 | 170.05 | 170.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 168.54 | 170.05 | 170.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 165.01 | 169.04 | 170.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 164.55 | 169.04 | 170.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 164.45 | 166.69 | 167.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 163.02 | 165.97 | 166.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 164.40 | 164.38 | 165.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 156.32 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 156.23 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 154.87 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 156.18 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-10 13:15:00 | 148.10 | 152.39 | 157.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 176 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 162.21 | 152.39 | 151.65 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 150.47 | 152.17 | 152.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 149.52 | 151.64 | 152.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 150.10 | 149.69 | 150.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 150.10 | 149.69 | 150.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 150.10 | 149.69 | 150.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 149.86 | 149.69 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 148.49 | 149.45 | 150.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:00:00 | 147.50 | 148.41 | 149.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 147.50 | 148.23 | 148.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 140.12 | 142.32 | 144.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 140.12 | 142.32 | 144.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 139.53 | 139.19 | 140.97 | SL hit (close>ema200) qty=0.50 sl=139.19 alert=retest2 |

### Cycle 178 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 144.13 | 141.82 | 141.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 144.64 | 142.39 | 141.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 145.20 | 145.78 | 144.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 145.21 | 145.78 | 144.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 144.93 | 145.61 | 144.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 144.74 | 145.61 | 144.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 144.94 | 145.39 | 144.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 144.94 | 145.39 | 144.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 144.38 | 145.19 | 144.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 144.38 | 145.19 | 144.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 145.29 | 145.21 | 144.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 145.44 | 145.21 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 151.86 | 152.11 | 150.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:30:00 | 152.49 | 152.29 | 150.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 150.61 | 152.10 | 151.02 | SL hit (close<static) qty=1.00 sl=150.62 alert=retest2 |

### Cycle 179 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 149.20 | 150.60 | 150.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 146.11 | 149.55 | 150.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 139.45 | 138.83 | 140.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 139.45 | 138.83 | 140.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 140.05 | 139.31 | 140.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 144.87 | 139.31 | 140.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 142.48 | 139.95 | 140.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 143.97 | 139.95 | 140.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 142.94 | 140.54 | 140.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 143.58 | 140.54 | 140.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 143.18 | 141.07 | 141.04 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 140.61 | 141.04 | 141.06 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 141.48 | 141.13 | 141.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 142.32 | 141.37 | 141.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 145.19 | 145.25 | 143.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:30:00 | 144.75 | 145.25 | 143.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 141.66 | 144.91 | 144.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 141.66 | 144.91 | 144.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 142.16 | 144.36 | 144.27 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 142.63 | 144.01 | 144.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 141.42 | 142.59 | 143.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 142.86 | 141.78 | 142.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 142.86 | 141.78 | 142.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 142.86 | 141.78 | 142.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:30:00 | 141.85 | 141.78 | 142.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 143.22 | 142.07 | 142.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 143.22 | 142.07 | 142.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 143.22 | 142.30 | 142.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 144.36 | 142.30 | 142.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 142.98 | 142.44 | 142.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 142.73 | 142.44 | 142.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 142.81 | 142.51 | 142.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 141.21 | 142.51 | 142.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 142.54 | 141.94 | 142.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 142.54 | 141.94 | 142.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 142.80 | 142.11 | 142.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 143.00 | 142.11 | 142.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 142.91 | 142.45 | 142.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 146.37 | 143.67 | 143.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 144.74 | 145.19 | 144.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 144.74 | 145.19 | 144.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 144.80 | 145.11 | 144.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 145.38 | 145.08 | 144.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 145.60 | 145.06 | 144.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 146.15 | 145.28 | 144.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 144.39 | 145.07 | 144.83 | SL hit (close<static) qty=1.00 sl=144.41 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 144.05 | 144.69 | 144.70 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 146.22 | 144.79 | 144.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 149.10 | 146.17 | 145.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 148.10 | 149.32 | 148.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 148.10 | 149.32 | 148.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 148.10 | 149.32 | 148.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 148.50 | 149.32 | 148.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 148.60 | 149.17 | 148.11 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 146.46 | 147.69 | 147.74 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 148.10 | 147.77 | 147.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 149.53 | 148.12 | 147.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 147.50 | 148.30 | 148.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 147.50 | 148.30 | 148.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 147.50 | 148.30 | 148.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 147.33 | 148.30 | 148.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 146.73 | 147.98 | 147.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 146.37 | 147.61 | 147.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 147.96 | 147.68 | 147.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 147.96 | 147.68 | 147.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 147.96 | 147.68 | 147.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 147.96 | 147.68 | 147.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 147.90 | 147.72 | 147.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 147.25 | 147.72 | 147.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 146.49 | 147.48 | 147.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 146.15 | 147.48 | 147.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 146.15 | 144.91 | 145.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 144.85 | 145.25 | 145.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 138.84 | 142.86 | 143.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 138.84 | 142.86 | 143.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 137.61 | 139.65 | 141.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 137.34 | 137.22 | 139.27 | SL hit (close>ema200) qty=0.50 sl=137.22 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 141.94 | 138.33 | 137.95 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 136.19 | 137.98 | 138.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 134.80 | 137.34 | 137.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 131.13 | 128.92 | 130.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 131.13 | 128.92 | 130.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 131.13 | 128.92 | 130.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 131.75 | 128.92 | 130.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 130.00 | 129.14 | 130.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 129.64 | 129.33 | 130.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:45:00 | 129.60 | 130.32 | 130.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 129.39 | 129.96 | 130.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 129.63 | 128.61 | 128.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 129.63 | 128.61 | 128.55 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 127.34 | 128.30 | 128.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 127.00 | 128.04 | 128.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 14:15:00 | 128.17 | 127.59 | 127.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 128.17 | 127.59 | 127.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 128.17 | 127.59 | 127.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 128.17 | 127.59 | 127.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 128.36 | 127.75 | 128.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 125.94 | 127.75 | 128.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 129.56 | 128.11 | 128.16 | SL hit (close>static) qty=1.00 sl=128.70 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 129.25 | 128.34 | 128.26 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 127.22 | 128.20 | 128.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 125.83 | 127.73 | 128.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 122.31 | 120.77 | 123.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 122.31 | 120.77 | 123.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 122.84 | 121.18 | 123.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 126.08 | 121.18 | 123.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 125.38 | 122.02 | 123.37 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 128.05 | 124.83 | 124.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 131.28 | 127.12 | 125.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 128.06 | 129.58 | 127.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 128.06 | 129.58 | 127.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 128.06 | 129.58 | 127.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 127.89 | 129.58 | 127.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 126.57 | 128.98 | 127.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 126.57 | 128.98 | 127.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 126.26 | 128.43 | 127.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 126.40 | 128.43 | 127.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 125.15 | 127.17 | 127.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 117.75 | 124.66 | 126.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 122.44 | 122.13 | 124.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 122.44 | 122.13 | 124.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 127.40 | 123.26 | 124.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 125.92 | 123.26 | 124.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 129.32 | 124.47 | 124.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 129.32 | 124.47 | 124.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 129.14 | 125.41 | 125.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 130.25 | 127.41 | 126.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 14:15:00 | 138.22 | 138.24 | 135.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:30:00 | 138.00 | 138.24 | 135.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 138.92 | 138.37 | 136.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 140.02 | 138.44 | 136.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 140.74 | 138.44 | 136.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 135.50 | 137.30 | 137.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 135.50 | 137.30 | 137.38 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 138.15 | 136.79 | 136.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 141.56 | 137.74 | 137.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 140.47 | 141.23 | 139.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 140.47 | 141.23 | 139.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 138.80 | 140.58 | 139.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 138.80 | 140.58 | 139.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 139.27 | 140.32 | 139.63 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 136.38 | 138.74 | 139.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 134.89 | 137.61 | 138.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 133.58 | 133.31 | 134.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 133.58 | 133.31 | 134.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 133.58 | 133.31 | 134.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 133.77 | 133.31 | 134.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 133.95 | 133.44 | 134.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 132.74 | 133.35 | 134.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:00:00 | 132.50 | 133.18 | 134.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 132.49 | 133.62 | 133.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 131.70 | 133.43 | 133.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 130.64 | 132.87 | 133.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 129.88 | 132.87 | 133.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 126.10 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 125.88 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 125.87 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 125.11 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 123.39 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 126.59 | 126.32 | 128.46 | SL hit (close>ema200) qty=0.50 sl=126.32 alert=retest2 |

### Cycle 202 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 126.43 | 120.22 | 119.81 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 119.80 | 121.02 | 121.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 118.99 | 120.62 | 120.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 118.34 | 117.93 | 119.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 118.34 | 117.93 | 119.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 118.34 | 117.93 | 119.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 118.81 | 117.93 | 119.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 118.80 | 118.10 | 119.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 118.05 | 118.10 | 119.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 124.06 | 120.14 | 119.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 124.06 | 120.14 | 119.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 126.29 | 121.37 | 120.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 124.48 | 125.21 | 123.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 124.48 | 125.21 | 123.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 124.48 | 125.21 | 123.08 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 117.09 | 122.02 | 122.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 116.51 | 119.73 | 121.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 119.17 | 118.92 | 120.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 119.17 | 118.92 | 120.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 119.17 | 118.92 | 120.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 118.77 | 118.96 | 120.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 118.85 | 118.96 | 120.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:00:00 | 118.65 | 118.90 | 120.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 125.03 | 120.95 | 120.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 125.03 | 120.95 | 120.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 125.64 | 121.89 | 121.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 120.72 | 123.19 | 122.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 120.72 | 123.19 | 122.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 120.72 | 123.19 | 122.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 120.72 | 123.19 | 122.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 120.01 | 122.56 | 122.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 120.10 | 122.56 | 122.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 119.75 | 121.57 | 121.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 117.05 | 120.14 | 120.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 119.56 | 116.83 | 118.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 119.56 | 116.83 | 118.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 119.56 | 116.83 | 118.46 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 120.03 | 119.22 | 119.18 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 115.97 | 118.69 | 118.96 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 120.00 | 119.02 | 118.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 120.80 | 119.63 | 119.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 120.70 | 120.97 | 120.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 120.70 | 120.97 | 120.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 120.85 | 120.99 | 120.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 120.57 | 120.99 | 120.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 120.67 | 120.93 | 120.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 124.41 | 120.93 | 120.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 136.85 | 134.64 | 133.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 132.80 | 134.96 | 135.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 132.68 | 133.66 | 134.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 134.70 | 133.79 | 134.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 134.70 | 133.79 | 134.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 134.70 | 133.79 | 134.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:45:00 | 135.07 | 133.79 | 134.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 133.66 | 133.76 | 134.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 133.25 | 133.79 | 134.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 133.12 | 133.66 | 134.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 135.70 | 133.94 | 134.10 | SL hit (close>static) qty=1.00 sl=134.77 alert=retest2 |

### Cycle 212 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 136.76 | 134.50 | 134.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 137.38 | 135.08 | 134.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 13:15:00 | 137.50 | 137.82 | 136.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 137.50 | 137.82 | 136.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 138.78 | 139.78 | 138.89 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 137.73 | 138.54 | 138.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 136.85 | 138.20 | 138.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 136.65 | 136.58 | 137.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:00:00 | 136.65 | 136.58 | 137.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 137.62 | 136.78 | 137.44 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 141.50 | 138.22 | 137.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 143.40 | 142.41 | 141.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 142.86 | 142.88 | 142.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 142.86 | 142.88 | 142.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 143.78 | 143.07 | 142.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:00:00 | 145.20 | 143.49 | 142.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 83.60 | 2024-04-15 09:15:00 | 80.10 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-04-12 15:15:00 | 82.70 | 2024-04-15 09:15:00 | 80.10 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-05-02 15:15:00 | 85.05 | 2024-05-03 12:15:00 | 84.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-05-13 09:15:00 | 79.35 | 2024-05-14 15:15:00 | 80.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-05-13 13:15:00 | 79.25 | 2024-05-14 15:15:00 | 80.40 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-05-18 09:15:00 | 81.95 | 2024-05-21 10:15:00 | 81.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-05-18 09:45:00 | 82.00 | 2024-05-21 10:15:00 | 81.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-05-21 13:00:00 | 81.90 | 2024-05-22 10:15:00 | 81.25 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-21 13:30:00 | 82.45 | 2024-05-22 10:15:00 | 81.25 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-05-22 14:45:00 | 82.35 | 2024-05-23 09:15:00 | 79.75 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-05-29 13:15:00 | 81.55 | 2024-05-30 09:15:00 | 80.25 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-05-29 15:15:00 | 81.50 | 2024-05-30 09:15:00 | 80.25 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-06-04 09:15:00 | 76.85 | 2024-06-04 11:15:00 | 73.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 76.85 | 2024-06-04 12:15:00 | 69.16 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-10 09:30:00 | 96.31 | 2024-07-10 10:15:00 | 92.78 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2024-08-09 14:15:00 | 97.52 | 2024-08-14 09:15:00 | 92.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 14:15:00 | 97.52 | 2024-08-19 09:15:00 | 94.35 | STOP_HIT | 0.50 | 3.25% |
| BUY | retest2 | 2024-09-03 12:45:00 | 113.50 | 2024-09-05 09:15:00 | 124.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-09 11:45:00 | 140.03 | 2024-10-10 09:15:00 | 145.80 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2024-11-05 11:30:00 | 137.20 | 2024-11-05 12:15:00 | 138.79 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-11-08 14:30:00 | 146.53 | 2024-11-11 09:15:00 | 144.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-11-08 15:15:00 | 146.25 | 2024-11-11 09:15:00 | 144.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-11-29 14:15:00 | 135.89 | 2024-12-10 14:15:00 | 139.96 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest2 | 2024-12-02 12:15:00 | 137.88 | 2024-12-10 14:15:00 | 139.96 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2024-12-17 09:45:00 | 139.03 | 2024-12-19 09:15:00 | 132.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:45:00 | 139.03 | 2024-12-19 15:15:00 | 133.70 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2025-01-09 11:45:00 | 122.34 | 2025-01-13 09:15:00 | 116.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 122.18 | 2025-01-13 09:15:00 | 116.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 122.40 | 2025-01-13 09:15:00 | 116.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:45:00 | 122.34 | 2025-01-14 09:15:00 | 115.49 | STOP_HIT | 0.50 | 5.60% |
| SELL | retest2 | 2025-01-09 12:15:00 | 122.18 | 2025-01-14 09:15:00 | 115.49 | STOP_HIT | 0.50 | 5.48% |
| SELL | retest2 | 2025-01-09 15:15:00 | 122.40 | 2025-01-14 09:15:00 | 115.49 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2025-01-23 14:30:00 | 114.47 | 2025-01-27 09:15:00 | 108.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 114.39 | 2025-01-27 09:15:00 | 108.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 114.47 | 2025-01-28 11:15:00 | 107.82 | STOP_HIT | 0.50 | 5.81% |
| SELL | retest2 | 2025-01-23 15:00:00 | 114.39 | 2025-01-28 11:15:00 | 107.82 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-02-18 09:15:00 | 98.41 | 2025-02-19 09:15:00 | 100.82 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-02-19 10:45:00 | 99.95 | 2025-02-19 11:15:00 | 101.40 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-02-24 10:15:00 | 101.11 | 2025-02-24 13:15:00 | 101.16 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-03-04 12:00:00 | 92.36 | 2025-03-05 09:15:00 | 94.61 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-03-04 15:15:00 | 92.00 | 2025-03-05 09:15:00 | 94.61 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-03-07 09:15:00 | 97.25 | 2025-03-10 09:15:00 | 93.39 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-03-13 11:30:00 | 88.85 | 2025-03-17 13:15:00 | 84.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-17 09:30:00 | 88.80 | 2025-03-17 13:15:00 | 84.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-13 11:30:00 | 88.85 | 2025-03-18 09:15:00 | 87.86 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2025-03-17 09:30:00 | 88.80 | 2025-03-18 09:15:00 | 87.86 | STOP_HIT | 0.50 | 1.06% |
| BUY | retest2 | 2025-03-26 09:15:00 | 99.60 | 2025-03-27 11:15:00 | 95.30 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2025-03-27 10:30:00 | 96.64 | 2025-03-27 11:15:00 | 95.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-04-01 11:30:00 | 95.12 | 2025-04-01 15:15:00 | 96.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-05 12:30:00 | 101.18 | 2025-05-08 10:15:00 | 101.59 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-05-05 15:15:00 | 101.20 | 2025-05-08 10:15:00 | 101.59 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-21 09:30:00 | 118.30 | 2025-05-29 09:15:00 | 130.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 142.42 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2025-06-18 14:15:00 | 141.83 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.79% |
| SELL | retest2 | 2025-06-18 15:15:00 | 142.15 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.55% |
| SELL | retest2 | 2025-06-20 12:15:00 | 142.05 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.62% |
| SELL | retest2 | 2025-06-23 13:00:00 | 139.74 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -9.40% |
| SELL | retest2 | 2025-06-23 14:15:00 | 139.61 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -9.51% |
| BUY | retest2 | 2025-07-02 09:15:00 | 159.50 | 2025-07-11 09:15:00 | 167.98 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2025-07-02 12:15:00 | 159.71 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | 4.09% |
| BUY | retest2 | 2025-07-02 13:15:00 | 159.40 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-07-02 14:30:00 | 161.58 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-07-08 09:15:00 | 173.23 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2025-07-08 11:30:00 | 169.36 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-08 13:30:00 | 169.90 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-07-08 14:15:00 | 169.06 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-11 09:15:00 | 170.20 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-07-15 11:15:00 | 172.21 | 2025-07-17 11:15:00 | 169.75 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-21 09:15:00 | 166.50 | 2025-07-22 13:15:00 | 170.69 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-07-22 09:45:00 | 168.10 | 2025-07-22 13:15:00 | 170.69 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-05 10:15:00 | 154.39 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-05 11:30:00 | 154.25 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-08-06 09:45:00 | 153.80 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-08-06 14:30:00 | 154.43 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-19 12:15:00 | 187.05 | 2025-08-26 10:15:00 | 186.90 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-08-19 14:15:00 | 187.00 | 2025-08-26 10:15:00 | 186.90 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-08-21 10:15:00 | 192.97 | 2025-08-26 10:15:00 | 186.90 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-09-12 12:45:00 | 180.54 | 2025-09-19 12:15:00 | 171.66 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-09-12 14:45:00 | 180.70 | 2025-09-19 12:15:00 | 171.81 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-09-15 09:45:00 | 179.92 | 2025-09-19 14:15:00 | 171.51 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2025-09-12 12:45:00 | 180.54 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-09-12 14:45:00 | 180.70 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-09-16 09:30:00 | 180.85 | 2025-09-22 09:15:00 | 170.92 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2025-09-15 09:45:00 | 179.92 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-09-16 09:30:00 | 180.85 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-09-17 10:30:00 | 176.94 | 2025-09-22 09:15:00 | 168.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 176.94 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-09-17 11:00:00 | 176.91 | 2025-09-22 09:15:00 | 168.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:00:00 | 176.91 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2025-09-17 11:45:00 | 176.52 | 2025-09-22 09:15:00 | 167.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:45:00 | 176.52 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-09-18 11:00:00 | 176.74 | 2025-09-22 09:15:00 | 167.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 176.74 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 2.00% |
| SELL | retest2 | 2025-09-23 11:00:00 | 171.66 | 2025-09-26 14:15:00 | 163.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 171.26 | 2025-09-26 14:15:00 | 162.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:15:00 | 171.87 | 2025-09-26 14:15:00 | 163.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:00:00 | 171.66 | 2025-10-01 11:15:00 | 160.63 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2025-09-24 09:15:00 | 171.26 | 2025-10-01 11:15:00 | 160.63 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2025-09-24 14:15:00 | 171.87 | 2025-10-01 11:15:00 | 160.63 | STOP_HIT | 0.50 | 6.54% |
| SELL | retest2 | 2025-10-29 10:15:00 | 164.55 | 2025-11-07 09:15:00 | 156.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 15:00:00 | 164.45 | 2025-11-07 09:15:00 | 156.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:15:00 | 163.02 | 2025-11-07 09:15:00 | 154.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 14:30:00 | 164.40 | 2025-11-07 09:15:00 | 156.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 10:15:00 | 164.55 | 2025-11-10 13:15:00 | 148.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 15:00:00 | 164.45 | 2025-11-10 13:15:00 | 148.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 09:15:00 | 163.02 | 2025-11-10 13:15:00 | 146.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 14:30:00 | 164.40 | 2025-11-10 13:15:00 | 147.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 14:00:00 | 147.50 | 2025-11-21 14:15:00 | 140.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 147.50 | 2025-11-21 14:15:00 | 140.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 14:00:00 | 147.50 | 2025-11-25 11:15:00 | 139.53 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-11-19 15:15:00 | 147.50 | 2025-11-25 11:15:00 | 139.53 | STOP_HIT | 0.50 | 5.40% |
| BUY | retest2 | 2025-12-02 14:30:00 | 152.49 | 2025-12-03 09:15:00 | 150.61 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 09:45:00 | 145.38 | 2025-12-26 14:15:00 | 144.39 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-26 12:15:00 | 145.60 | 2025-12-26 14:15:00 | 144.39 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-26 13:00:00 | 146.15 | 2025-12-26 14:15:00 | 144.39 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-06 10:15:00 | 146.15 | 2026-01-09 09:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:45:00 | 146.15 | 2026-01-09 09:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 144.85 | 2026-01-12 09:15:00 | 137.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:15:00 | 146.15 | 2026-01-12 15:15:00 | 137.34 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-01-07 14:45:00 | 146.15 | 2026-01-12 15:15:00 | 137.34 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-01-08 09:15:00 | 144.85 | 2026-01-12 15:15:00 | 137.34 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-22 11:30:00 | 129.64 | 2026-01-28 15:15:00 | 129.63 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-01-23 09:45:00 | 129.60 | 2026-01-28 15:15:00 | 129.63 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-01-23 10:30:00 | 129.39 | 2026-01-28 15:15:00 | 129.63 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-01-30 09:15:00 | 125.94 | 2026-01-30 09:15:00 | 129.56 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-02-12 12:30:00 | 140.02 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-02-12 13:15:00 | 140.74 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-02-25 12:45:00 | 132.74 | 2026-03-02 09:15:00 | 126.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 132.50 | 2026-03-02 09:15:00 | 125.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 14:15:00 | 132.49 | 2026-03-02 09:15:00 | 125.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 131.70 | 2026-03-02 09:15:00 | 125.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:15:00 | 129.88 | 2026-03-02 09:15:00 | 123.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 132.74 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2026-02-25 14:00:00 | 132.50 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2026-02-26 14:15:00 | 132.49 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-27 09:15:00 | 131.70 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-02-27 10:15:00 | 129.88 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2026-03-17 09:15:00 | 118.05 | 2026-03-18 09:15:00 | 124.06 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2026-03-24 10:30:00 | 118.77 | 2026-03-25 09:15:00 | 125.03 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest2 | 2026-03-24 11:15:00 | 118.85 | 2026-03-25 09:15:00 | 125.03 | STOP_HIT | 1.00 | -5.20% |
| SELL | retest2 | 2026-03-24 12:00:00 | 118.65 | 2026-03-25 09:15:00 | 125.03 | STOP_HIT | 1.00 | -5.38% |
| BUY | retest2 | 2026-04-08 09:15:00 | 124.41 | 2026-04-16 09:15:00 | 136.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 133.25 | 2026-04-23 11:15:00 | 135.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-23 10:00:00 | 133.12 | 2026-04-23 11:15:00 | 135.70 | STOP_HIT | 1.00 | -1.94% |
