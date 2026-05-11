# PCBL Chemical Ltd. (PCBL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 306.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 203 |
| ALERT1 | 135 |
| ALERT2 | 134 |
| ALERT2_SKIP | 93 |
| ALERT3 | 334 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 101 |
| PARTIAL | 28 |
| TARGET_HIT | 12 |
| STOP_HIT | 86 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 7
- **Winners / losers:** 77 / 49
- **Target hits / Stop hits / Partials:** 12 / 86 / 28
- **Avg / median % per leg:** 2.16% / 2.32%
- **Sum % (uncompounded):** 272.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 18 | 41.9% | 10 | 33 | 0 | 1.82% | 78.4% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.90% | -7.6% |
| BUY @ 3rd Alert (retest2) | 39 | 18 | 46.2% | 10 | 29 | 0 | 2.20% | 86.0% |
| SELL (all) | 83 | 59 | 71.1% | 2 | 53 | 28 | 2.33% | 193.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 83 | 59 | 71.1% | 2 | 53 | 28 | 2.33% | 193.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.90% | -7.6% |
| retest2 (combined) | 122 | 77 | 63.1% | 12 | 82 | 28 | 2.29% | 279.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 130.90 | 129.46 | 129.39 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 128.90 | 129.64 | 129.65 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 132.65 | 130.24 | 129.93 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 128.95 | 130.86 | 130.89 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 133.00 | 130.99 | 130.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 09:15:00 | 134.15 | 131.95 | 131.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 12:15:00 | 134.50 | 135.29 | 134.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 13:15:00 | 134.05 | 135.04 | 134.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 134.05 | 135.04 | 134.04 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 11:15:00 | 133.65 | 134.51 | 134.60 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 10:15:00 | 135.65 | 134.38 | 134.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 11:15:00 | 137.85 | 136.31 | 135.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 15:15:00 | 137.45 | 138.35 | 137.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 15:15:00 | 137.45 | 138.35 | 137.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 137.45 | 138.35 | 137.46 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 12:15:00 | 157.00 | 158.02 | 158.03 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 159.45 | 158.27 | 158.11 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 157.15 | 158.07 | 158.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 155.40 | 157.18 | 157.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 155.50 | 155.15 | 156.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 155.50 | 155.15 | 156.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 155.50 | 155.15 | 156.20 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 13:15:00 | 157.80 | 156.79 | 156.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 158.80 | 157.19 | 156.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 09:15:00 | 157.75 | 157.82 | 157.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 157.75 | 157.82 | 157.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 157.75 | 157.82 | 157.52 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 12:15:00 | 162.30 | 163.53 | 163.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 13:15:00 | 162.00 | 163.22 | 163.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 163.00 | 161.11 | 161.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 163.00 | 161.11 | 161.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 163.00 | 161.11 | 161.79 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 15:15:00 | 163.80 | 162.23 | 162.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 174.20 | 164.63 | 163.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 14:15:00 | 168.50 | 169.06 | 166.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 14:15:00 | 168.50 | 169.06 | 166.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 168.50 | 169.06 | 166.35 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 13:15:00 | 163.70 | 165.40 | 165.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 09:15:00 | 159.95 | 163.61 | 164.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 158.00 | 157.38 | 159.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 159.55 | 157.81 | 159.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 159.55 | 157.81 | 159.23 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 161.00 | 158.30 | 158.00 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 157.25 | 158.34 | 158.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 11:15:00 | 155.45 | 157.35 | 157.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 157.10 | 156.09 | 156.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 157.10 | 156.09 | 156.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 157.10 | 156.09 | 156.96 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 09:15:00 | 160.45 | 157.55 | 157.27 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 09:15:00 | 154.95 | 157.49 | 157.65 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 158.50 | 157.18 | 157.02 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 155.75 | 157.54 | 157.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 154.55 | 156.95 | 157.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 155.80 | 155.49 | 156.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 154.20 | 155.27 | 156.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 154.20 | 155.27 | 156.04 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 161.05 | 155.46 | 154.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 10:15:00 | 161.90 | 156.75 | 155.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 15:15:00 | 157.65 | 157.94 | 156.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 09:15:00 | 155.40 | 157.44 | 156.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 155.40 | 157.44 | 156.52 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 13:15:00 | 154.00 | 155.83 | 155.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 14:15:00 | 153.70 | 154.52 | 155.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 154.80 | 154.43 | 154.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 154.80 | 154.43 | 154.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 154.80 | 154.43 | 154.94 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 15:15:00 | 155.70 | 155.17 | 155.15 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 154.60 | 155.04 | 155.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 153.80 | 154.79 | 154.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 154.80 | 154.30 | 154.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 154.80 | 154.30 | 154.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 154.80 | 154.30 | 154.60 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 156.10 | 154.46 | 154.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 10:15:00 | 157.00 | 154.97 | 154.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 156.75 | 157.44 | 156.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 13:15:00 | 156.75 | 157.44 | 156.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 156.75 | 157.44 | 156.50 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 12:15:00 | 170.25 | 172.25 | 172.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 15:15:00 | 169.20 | 170.98 | 171.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 176.20 | 171.23 | 171.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 176.20 | 171.23 | 171.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 176.20 | 171.23 | 171.31 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 178.45 | 172.68 | 171.96 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 163.90 | 171.76 | 172.55 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 169.10 | 167.49 | 167.46 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 167.20 | 167.89 | 167.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 162.05 | 165.74 | 166.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 14:15:00 | 164.35 | 164.32 | 165.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 163.75 | 164.20 | 165.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 163.75 | 164.20 | 165.27 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 167.15 | 164.04 | 163.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 11:15:00 | 169.90 | 168.36 | 167.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 168.00 | 168.53 | 167.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 15:15:00 | 168.00 | 168.63 | 167.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 168.00 | 168.63 | 167.62 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 197.25 | 202.84 | 203.04 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 10:15:00 | 206.25 | 203.35 | 203.07 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 13:15:00 | 202.60 | 204.64 | 204.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 192.25 | 201.72 | 203.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 13:15:00 | 189.55 | 189.49 | 193.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 191.50 | 188.89 | 190.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 191.50 | 188.89 | 190.51 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 192.50 | 191.46 | 191.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 12:15:00 | 195.50 | 193.15 | 192.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 13:15:00 | 197.70 | 198.71 | 196.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 14:15:00 | 198.70 | 199.33 | 198.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 198.70 | 199.33 | 198.29 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 196.25 | 197.85 | 198.01 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 201.50 | 198.42 | 198.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 203.15 | 199.90 | 198.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 10:15:00 | 202.45 | 202.53 | 200.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 11:15:00 | 202.20 | 202.46 | 201.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 202.20 | 202.46 | 201.09 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 267.65 | 272.04 | 272.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 09:15:00 | 266.85 | 269.73 | 270.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 11:15:00 | 254.85 | 253.99 | 256.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 255.35 | 254.55 | 255.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 255.35 | 254.55 | 255.94 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 264.95 | 258.02 | 257.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 268.95 | 262.92 | 260.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 15:15:00 | 264.35 | 264.47 | 262.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 261.25 | 263.83 | 262.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 261.25 | 263.83 | 262.25 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 13:15:00 | 259.45 | 261.25 | 261.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 12:15:00 | 257.95 | 259.79 | 260.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 246.95 | 246.41 | 250.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 249.90 | 247.26 | 250.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 249.90 | 247.26 | 250.09 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 256.60 | 250.97 | 250.64 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 10:15:00 | 252.95 | 253.70 | 253.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 13:15:00 | 252.30 | 253.13 | 253.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-01 09:15:00 | 254.15 | 252.85 | 253.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 254.15 | 252.85 | 253.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 254.15 | 252.85 | 253.18 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 256.25 | 253.53 | 253.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 09:15:00 | 256.75 | 255.17 | 254.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 253.45 | 254.83 | 254.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 253.45 | 254.83 | 254.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 253.45 | 254.83 | 254.33 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 15:15:00 | 254.80 | 255.23 | 255.23 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 262.75 | 256.73 | 255.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 13:15:00 | 266.35 | 263.16 | 261.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 10:15:00 | 263.60 | 264.31 | 262.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 12:15:00 | 265.50 | 265.83 | 264.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 265.50 | 265.83 | 264.56 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 307.45 | 310.72 | 310.81 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 09:15:00 | 319.00 | 312.37 | 311.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 327.05 | 322.24 | 319.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 14:15:00 | 323.15 | 324.27 | 321.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 325.90 | 324.42 | 322.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 325.90 | 324.42 | 322.09 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 09:15:00 | 317.65 | 321.08 | 321.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 10:15:00 | 313.50 | 319.56 | 320.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 318.80 | 317.08 | 318.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 14:15:00 | 318.80 | 317.08 | 318.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 318.80 | 317.08 | 318.89 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 320.75 | 318.77 | 318.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 323.20 | 320.49 | 319.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 13:15:00 | 323.40 | 324.57 | 322.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 13:15:00 | 323.40 | 324.57 | 322.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 13:15:00 | 323.40 | 324.57 | 322.24 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 11:15:00 | 318.70 | 323.17 | 323.41 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 15:15:00 | 327.00 | 323.94 | 323.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 09:15:00 | 339.00 | 326.95 | 325.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 323.90 | 331.17 | 328.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 323.90 | 331.17 | 328.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 323.90 | 331.17 | 328.98 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 319.30 | 326.59 | 327.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 303.65 | 318.90 | 322.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 301.65 | 299.60 | 305.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 303.65 | 300.41 | 304.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 303.65 | 300.41 | 304.96 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 308.50 | 305.51 | 305.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 319.00 | 308.83 | 306.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 318.80 | 320.65 | 315.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 12:15:00 | 315.90 | 318.74 | 315.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 315.90 | 318.74 | 315.68 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 15:15:00 | 312.85 | 314.98 | 315.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 09:15:00 | 310.35 | 314.06 | 314.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 307.35 | 307.22 | 310.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 307.35 | 307.22 | 310.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 307.35 | 307.22 | 310.27 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 15:15:00 | 295.50 | 292.18 | 292.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 296.30 | 293.00 | 292.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 293.75 | 294.77 | 293.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 293.75 | 294.77 | 293.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 293.75 | 294.77 | 293.54 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 13:15:00 | 290.30 | 292.60 | 292.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 15:15:00 | 289.25 | 291.52 | 292.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 15:15:00 | 277.90 | 277.38 | 281.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 249.30 | 239.83 | 245.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 249.30 | 239.83 | 245.96 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 255.20 | 249.23 | 248.94 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 09:15:00 | 243.90 | 248.23 | 248.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 242.05 | 247.00 | 247.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 13:15:00 | 246.10 | 245.81 | 247.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 13:15:00 | 246.10 | 245.81 | 247.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 246.10 | 245.81 | 247.07 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 250.55 | 247.57 | 247.47 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 15:15:00 | 246.80 | 247.64 | 247.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 242.90 | 246.69 | 247.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 246.40 | 245.97 | 246.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 246.40 | 245.97 | 246.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 246.40 | 245.97 | 246.73 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 254.00 | 248.05 | 247.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 255.80 | 250.67 | 249.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 15:15:00 | 260.75 | 260.97 | 258.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 15:15:00 | 266.75 | 268.12 | 265.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 266.75 | 268.12 | 265.36 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 275.90 | 278.89 | 278.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 15:15:00 | 274.60 | 276.47 | 277.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 282.05 | 277.59 | 278.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 282.05 | 277.59 | 278.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 282.05 | 277.59 | 278.01 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 11:15:00 | 280.55 | 278.70 | 278.47 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 11:15:00 | 276.50 | 278.25 | 278.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 275.35 | 276.83 | 277.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 10:15:00 | 268.25 | 266.64 | 268.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 11:00:00 | 268.25 | 266.64 | 268.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 262.00 | 260.41 | 262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:30:00 | 263.90 | 260.41 | 262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 11:15:00 | 263.65 | 261.39 | 262.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 11:30:00 | 263.50 | 261.39 | 262.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 262.90 | 261.69 | 262.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:30:00 | 263.05 | 261.69 | 262.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 13:15:00 | 262.75 | 261.90 | 262.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 14:15:00 | 263.30 | 261.90 | 262.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 263.90 | 262.30 | 262.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 263.90 | 262.30 | 262.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 263.05 | 262.45 | 262.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:15:00 | 262.85 | 262.45 | 262.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 262.35 | 262.54 | 262.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:30:00 | 262.65 | 262.54 | 262.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 267.00 | 263.43 | 263.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 268.50 | 265.77 | 264.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 13:15:00 | 265.45 | 266.06 | 265.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 13:15:00 | 265.45 | 266.06 | 265.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 265.45 | 266.06 | 265.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 13:30:00 | 265.80 | 266.06 | 265.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 264.75 | 265.80 | 265.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 15:00:00 | 264.75 | 265.80 | 265.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 265.45 | 265.73 | 265.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:15:00 | 266.95 | 265.73 | 265.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 266.50 | 265.88 | 265.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:15:00 | 267.50 | 265.88 | 265.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:15:00 | 267.85 | 266.05 | 265.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:30:00 | 267.25 | 266.91 | 266.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 269.75 | 266.89 | 266.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 265.85 | 266.68 | 266.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 265.85 | 266.68 | 266.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 267.80 | 266.91 | 266.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:45:00 | 266.50 | 266.91 | 266.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 273.25 | 275.43 | 273.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 273.20 | 275.43 | 273.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 274.35 | 275.22 | 273.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 269.90 | 272.51 | 272.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 269.90 | 272.51 | 272.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 264.15 | 268.14 | 269.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 261.90 | 257.99 | 261.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 261.90 | 257.99 | 261.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 261.90 | 257.99 | 261.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 261.55 | 257.99 | 261.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 261.50 | 258.69 | 261.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:15:00 | 262.60 | 258.69 | 261.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 262.50 | 259.45 | 261.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 12:45:00 | 261.25 | 259.90 | 261.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:30:00 | 260.40 | 259.98 | 261.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 257.80 | 260.97 | 261.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 248.19 | 253.72 | 256.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 247.38 | 253.72 | 256.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 244.91 | 253.72 | 256.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 13:15:00 | 254.35 | 253.11 | 255.37 | SL hit (close>ema200) qty=0.50 sl=253.11 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 255.80 | 252.04 | 252.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 259.95 | 254.58 | 253.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 256.40 | 256.43 | 254.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 14:30:00 | 256.70 | 256.43 | 254.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 264.70 | 265.47 | 264.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:45:00 | 264.65 | 265.47 | 264.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 262.00 | 264.78 | 264.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 260.20 | 264.78 | 264.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 259.60 | 263.74 | 263.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 257.35 | 263.74 | 263.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 260.40 | 263.07 | 263.31 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 265.30 | 263.75 | 263.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 267.35 | 264.80 | 264.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 10:15:00 | 264.45 | 264.73 | 264.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 10:15:00 | 264.45 | 264.73 | 264.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 264.45 | 264.73 | 264.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 263.80 | 264.73 | 264.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 264.10 | 264.61 | 264.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 264.60 | 264.61 | 264.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 264.10 | 264.50 | 264.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:30:00 | 263.25 | 264.50 | 264.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 265.60 | 264.72 | 264.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:15:00 | 265.35 | 264.72 | 264.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 259.00 | 263.58 | 263.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 15:15:00 | 258.70 | 262.60 | 263.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 12:15:00 | 247.85 | 246.84 | 250.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 247.85 | 246.84 | 250.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 247.85 | 246.84 | 250.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 243.15 | 248.14 | 250.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-03 10:15:00 | 230.99 | 235.65 | 238.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-03 15:15:00 | 234.90 | 234.61 | 237.06 | SL hit (close>ema200) qty=0.50 sl=234.61 alert=retest2 |

### Cycle 71 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 231.20 | 228.17 | 227.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 232.40 | 229.49 | 228.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 11:15:00 | 244.90 | 245.66 | 242.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:00:00 | 244.90 | 245.66 | 242.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 243.87 | 245.09 | 243.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 243.87 | 245.09 | 243.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 244.49 | 244.84 | 243.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 245.16 | 244.84 | 243.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 244.36 | 244.72 | 243.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:30:00 | 245.00 | 244.72 | 243.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 252.25 | 251.50 | 249.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 253.00 | 251.99 | 250.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-25 09:15:00 | 278.30 | 271.39 | 266.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 264.99 | 267.47 | 267.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 09:15:00 | 262.46 | 265.65 | 266.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 259.13 | 258.61 | 261.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 259.13 | 258.61 | 261.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 259.13 | 258.61 | 261.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 258.62 | 258.61 | 261.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 257.75 | 256.68 | 258.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 256.10 | 256.68 | 258.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 255.20 | 256.38 | 257.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:30:00 | 253.00 | 255.46 | 257.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 253.15 | 255.28 | 256.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 10:45:00 | 254.20 | 254.61 | 256.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:15:00 | 254.20 | 254.60 | 255.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 256.70 | 254.92 | 255.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 257.30 | 254.92 | 255.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 257.50 | 255.44 | 255.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 257.50 | 255.44 | 255.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 258.80 | 256.36 | 256.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 258.80 | 256.36 | 256.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 261.20 | 257.81 | 257.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 267.65 | 268.33 | 265.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:00:00 | 267.65 | 268.33 | 265.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 267.00 | 268.07 | 265.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:00:00 | 268.40 | 267.85 | 265.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 262.05 | 266.76 | 265.91 | SL hit (close<static) qty=1.00 sl=264.40 alert=retest2 |

### Cycle 74 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 262.10 | 265.01 | 265.21 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 278.25 | 266.33 | 265.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 282.40 | 274.30 | 270.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 275.85 | 276.23 | 272.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 13:00:00 | 275.85 | 276.23 | 272.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 275.05 | 275.82 | 273.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 276.00 | 275.82 | 273.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:00:00 | 282.90 | 277.24 | 274.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 277.75 | 285.36 | 285.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 276.00 | 283.49 | 284.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 276.00 | 283.49 | 284.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 275.00 | 278.90 | 280.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 280.60 | 278.01 | 279.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 280.60 | 278.01 | 279.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 280.60 | 278.01 | 279.13 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 283.45 | 280.08 | 279.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 289.95 | 282.05 | 280.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 287.60 | 287.97 | 285.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 15:00:00 | 287.60 | 287.97 | 285.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 290.70 | 291.03 | 289.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:30:00 | 290.60 | 291.03 | 289.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 289.75 | 290.78 | 289.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:00:00 | 289.75 | 290.78 | 289.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 299.65 | 292.55 | 290.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:30:00 | 289.45 | 292.55 | 290.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 390.10 | 391.22 | 379.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 395.80 | 391.22 | 379.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 392.00 | 387.96 | 383.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 393.65 | 387.96 | 383.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:45:00 | 395.60 | 390.66 | 385.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 395.00 | 397.77 | 394.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 390.60 | 397.77 | 394.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 386.25 | 395.47 | 393.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:15:00 | 386.30 | 395.47 | 393.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 388.75 | 394.12 | 392.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:30:00 | 390.60 | 392.22 | 392.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 391.80 | 392.14 | 392.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 391.80 | 392.14 | 392.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 386.50 | 391.01 | 391.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 09:15:00 | 392.55 | 382.76 | 385.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 392.55 | 382.76 | 385.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 392.55 | 382.76 | 385.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:45:00 | 389.15 | 382.76 | 385.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 393.15 | 384.84 | 386.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:30:00 | 396.60 | 384.84 | 386.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 394.45 | 388.21 | 387.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 399.60 | 392.01 | 389.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 419.50 | 421.80 | 415.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 419.50 | 421.80 | 415.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 419.50 | 421.80 | 415.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 415.40 | 421.80 | 415.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 439.85 | 439.25 | 432.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 14:15:00 | 442.95 | 439.25 | 432.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-23 13:15:00 | 487.25 | 460.13 | 447.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 492.40 | 498.62 | 498.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 487.20 | 493.64 | 496.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 492.85 | 492.75 | 495.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 11:30:00 | 493.15 | 492.75 | 495.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 486.45 | 480.42 | 484.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 485.95 | 480.42 | 484.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 487.15 | 481.77 | 484.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 487.15 | 481.77 | 484.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 485.00 | 482.00 | 483.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 486.05 | 482.00 | 483.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 487.20 | 483.04 | 484.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 487.20 | 483.04 | 484.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 489.50 | 484.33 | 484.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 492.90 | 484.33 | 484.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 492.25 | 485.92 | 485.43 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 483.40 | 485.55 | 485.63 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 490.25 | 486.49 | 486.05 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 482.05 | 485.75 | 486.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 15:15:00 | 481.00 | 483.93 | 485.16 | Break + close below crossover candle low |

### Cycle 85 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 510.30 | 484.50 | 484.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 519.85 | 502.78 | 494.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 501.60 | 502.54 | 495.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 15:00:00 | 501.60 | 502.54 | 495.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 510.00 | 517.33 | 509.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 510.00 | 517.33 | 509.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 509.50 | 515.77 | 509.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 503.70 | 515.77 | 509.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 511.00 | 514.81 | 509.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 517.00 | 512.98 | 509.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:30:00 | 523.15 | 515.15 | 511.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 14:45:00 | 519.95 | 515.84 | 513.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-27 13:15:00 | 568.70 | 548.75 | 540.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 554.45 | 561.65 | 561.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 545.00 | 556.20 | 559.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 560.00 | 556.80 | 558.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 560.00 | 556.80 | 558.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 560.00 | 556.80 | 558.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 560.00 | 556.80 | 558.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 551.15 | 555.67 | 558.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:45:00 | 542.95 | 551.73 | 555.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 515.80 | 543.10 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:30:00 | 525.60 | 543.10 | 551.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 14:15:00 | 499.32 | 520.99 | 535.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-08 09:15:00 | 488.66 | 514.79 | 530.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 87 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 518.00 | 517.37 | 517.35 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 514.25 | 516.75 | 517.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 12:15:00 | 511.95 | 515.32 | 516.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 503.25 | 498.42 | 502.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 503.25 | 498.42 | 502.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 503.25 | 498.42 | 502.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 503.25 | 498.42 | 502.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 502.00 | 499.14 | 502.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 507.45 | 499.14 | 502.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 502.95 | 499.90 | 502.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 506.50 | 499.90 | 502.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 502.40 | 500.40 | 502.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:30:00 | 501.80 | 500.40 | 502.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 497.10 | 499.74 | 502.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:45:00 | 493.35 | 498.17 | 500.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 468.68 | 485.87 | 491.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 10:15:00 | 486.90 | 486.08 | 491.43 | SL hit (close>ema200) qty=0.50 sl=486.08 alert=retest2 |

### Cycle 89 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 445.40 | 442.61 | 442.46 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 438.85 | 442.47 | 442.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 430.45 | 438.70 | 440.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 433.35 | 422.65 | 428.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 433.35 | 422.65 | 428.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 433.35 | 422.65 | 428.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 433.35 | 422.65 | 428.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 424.70 | 423.06 | 428.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 412.25 | 423.06 | 428.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 429.80 | 422.51 | 422.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 429.80 | 422.51 | 422.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 438.05 | 426.83 | 424.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 436.10 | 436.55 | 431.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 436.10 | 436.55 | 431.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 435.50 | 436.34 | 431.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 431.50 | 436.34 | 431.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 429.65 | 434.48 | 431.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 429.65 | 434.48 | 431.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 432.55 | 434.10 | 431.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 430.00 | 434.10 | 431.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 430.95 | 433.47 | 431.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 430.95 | 433.47 | 431.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 431.00 | 432.97 | 431.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 429.70 | 432.97 | 431.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 432.00 | 432.78 | 431.46 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 419.90 | 429.17 | 429.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 417.60 | 422.51 | 426.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 421.50 | 417.67 | 421.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 421.50 | 417.67 | 421.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 421.50 | 417.67 | 421.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:15:00 | 422.20 | 417.67 | 421.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 416.95 | 417.53 | 420.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 414.00 | 417.34 | 420.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 393.30 | 406.41 | 413.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 400.40 | 394.92 | 402.87 | SL hit (close>ema200) qty=0.50 sl=394.92 alert=retest2 |

### Cycle 93 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 396.50 | 396.14 | 396.10 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 389.10 | 394.73 | 395.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 388.00 | 393.38 | 394.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 387.60 | 387.49 | 390.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 387.60 | 387.49 | 390.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 387.60 | 387.49 | 390.21 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 394.00 | 391.29 | 391.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 395.40 | 392.11 | 391.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 393.65 | 393.90 | 392.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 12:15:00 | 393.65 | 393.90 | 392.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 393.65 | 393.90 | 392.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:00:00 | 393.65 | 393.90 | 392.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 393.05 | 393.73 | 392.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:45:00 | 393.25 | 393.73 | 392.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 394.00 | 393.92 | 392.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 398.90 | 393.92 | 392.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:00:00 | 395.95 | 394.32 | 393.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:00:00 | 395.20 | 394.66 | 393.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 13:00:00 | 395.25 | 394.78 | 393.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-27 12:15:00 | 435.55 | 417.05 | 406.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 470.00 | 477.60 | 478.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 463.50 | 472.50 | 475.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 470.85 | 467.74 | 471.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 470.85 | 467.74 | 471.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 470.85 | 467.74 | 471.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 472.25 | 467.74 | 471.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 469.20 | 468.03 | 471.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 474.15 | 468.03 | 471.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 472.05 | 468.83 | 471.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 473.85 | 468.83 | 471.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 472.60 | 469.59 | 471.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:45:00 | 474.00 | 469.59 | 471.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 474.85 | 470.64 | 471.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 474.85 | 470.64 | 471.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 476.90 | 471.89 | 472.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 476.90 | 471.89 | 472.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 486.45 | 474.80 | 473.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 15:15:00 | 487.90 | 479.37 | 475.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 487.00 | 488.60 | 483.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:15:00 | 492.25 | 488.60 | 483.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 482.95 | 487.47 | 483.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 482.95 | 487.47 | 483.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 486.30 | 487.24 | 483.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 11:15:00 | 491.10 | 487.24 | 483.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:00:00 | 489.15 | 488.12 | 484.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 473.60 | 484.35 | 484.07 | SL hit (close<static) qty=1.00 sl=482.50 alert=retest2 |

### Cycle 98 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 475.30 | 482.54 | 483.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 464.20 | 475.57 | 478.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 459.70 | 458.73 | 462.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 459.70 | 458.73 | 462.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 459.70 | 458.73 | 462.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 461.55 | 458.73 | 462.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 462.95 | 459.58 | 462.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 462.90 | 459.58 | 462.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 459.90 | 459.64 | 462.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:45:00 | 460.10 | 459.64 | 462.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 462.85 | 460.28 | 462.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:45:00 | 461.80 | 460.28 | 462.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 461.85 | 460.60 | 462.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 14:45:00 | 458.65 | 460.36 | 462.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 15:15:00 | 457.90 | 460.36 | 462.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 457.85 | 459.88 | 461.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:00:00 | 457.45 | 458.26 | 460.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 462.80 | 459.17 | 460.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 462.80 | 459.17 | 460.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 462.65 | 459.86 | 460.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 458.75 | 459.86 | 460.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:00:00 | 460.15 | 460.59 | 460.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 435.72 | 453.20 | 457.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 435.00 | 453.20 | 457.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 434.96 | 453.20 | 457.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 434.58 | 453.20 | 457.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 435.81 | 453.20 | 457.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 437.14 | 453.20 | 457.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 446.75 | 444.60 | 450.65 | SL hit (close>ema200) qty=0.50 sl=444.60 alert=retest2 |

### Cycle 99 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 458.70 | 452.79 | 452.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 465.25 | 457.68 | 455.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 464.65 | 466.33 | 462.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:45:00 | 464.20 | 466.33 | 462.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 463.40 | 465.75 | 462.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 463.40 | 465.75 | 462.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 461.35 | 464.87 | 462.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 453.95 | 464.87 | 462.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 455.75 | 463.04 | 462.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:15:00 | 452.20 | 463.04 | 462.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 441.65 | 458.76 | 460.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 439.10 | 451.85 | 456.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 449.55 | 443.96 | 450.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 449.55 | 443.96 | 450.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 449.55 | 443.96 | 450.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 450.50 | 443.96 | 450.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 450.15 | 445.19 | 450.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 451.50 | 445.19 | 450.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 443.00 | 444.76 | 449.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:45:00 | 440.90 | 443.92 | 449.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 14:15:00 | 418.85 | 431.62 | 439.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-09 14:15:00 | 396.81 | 411.23 | 423.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 101 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 369.25 | 364.41 | 363.79 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 358.80 | 364.72 | 364.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 356.80 | 363.14 | 364.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 362.00 | 361.19 | 362.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 362.00 | 361.19 | 362.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 363.90 | 361.73 | 362.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 365.00 | 361.73 | 362.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 372.00 | 363.79 | 363.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 373.50 | 368.36 | 366.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 375.45 | 376.19 | 372.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 375.45 | 376.19 | 372.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 365.85 | 374.12 | 371.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 365.85 | 374.12 | 371.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 369.00 | 373.10 | 371.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 353.00 | 373.10 | 371.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 346.05 | 367.69 | 369.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 345.80 | 363.31 | 366.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 351.10 | 348.63 | 355.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:45:00 | 350.35 | 348.63 | 355.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 350.90 | 350.40 | 354.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 356.00 | 350.40 | 354.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 365.00 | 353.32 | 355.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 365.00 | 353.32 | 355.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 360.65 | 354.79 | 355.98 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 361.05 | 356.72 | 356.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 363.65 | 358.11 | 357.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 363.90 | 366.21 | 362.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 14:15:00 | 363.90 | 366.21 | 362.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 363.90 | 366.21 | 362.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 363.90 | 366.21 | 362.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 364.95 | 365.95 | 363.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 367.05 | 365.95 | 363.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-03 14:15:00 | 403.76 | 390.20 | 382.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 398.70 | 409.79 | 410.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 398.20 | 407.48 | 409.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 373.60 | 372.02 | 382.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 376.10 | 372.02 | 382.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 379.15 | 373.49 | 380.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 379.15 | 373.49 | 380.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 380.10 | 375.42 | 380.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:45:00 | 372.85 | 378.73 | 380.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 354.21 | 367.14 | 373.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 370.25 | 367.33 | 372.14 | SL hit (close>ema200) qty=0.50 sl=367.33 alert=retest2 |

### Cycle 107 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 373.85 | 371.23 | 371.12 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 15:15:00 | 368.00 | 371.11 | 371.12 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 380.00 | 372.89 | 371.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 387.00 | 375.71 | 373.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 382.10 | 382.81 | 378.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 12:00:00 | 387.85 | 383.85 | 379.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 12:30:00 | 387.85 | 384.54 | 380.53 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 13:45:00 | 389.35 | 385.83 | 381.48 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 382.85 | 391.03 | 387.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 382.85 | 391.03 | 387.91 | SL hit (close<ema400) qty=1.00 sl=387.91 alert=retest1 |

### Cycle 110 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 384.05 | 386.02 | 386.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 382.60 | 385.33 | 385.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 386.70 | 385.61 | 385.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 386.70 | 385.61 | 385.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 386.70 | 385.61 | 385.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 388.35 | 385.61 | 385.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 386.25 | 385.73 | 385.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:15:00 | 387.00 | 385.73 | 385.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 387.20 | 386.03 | 386.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:30:00 | 387.60 | 386.03 | 386.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 387.45 | 386.31 | 386.20 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 374.35 | 384.05 | 385.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 371.25 | 381.49 | 383.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 375.25 | 375.02 | 379.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 375.25 | 375.02 | 379.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 359.20 | 355.37 | 360.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 362.35 | 355.37 | 360.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 362.35 | 356.76 | 360.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:15:00 | 365.75 | 356.76 | 360.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 362.90 | 357.99 | 360.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:45:00 | 365.40 | 357.99 | 360.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 366.15 | 360.50 | 361.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 366.15 | 360.50 | 361.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 373.45 | 364.00 | 362.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 377.80 | 370.40 | 366.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 385.20 | 386.19 | 380.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 385.20 | 386.19 | 380.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 386.10 | 387.79 | 385.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:15:00 | 380.15 | 387.79 | 385.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 380.15 | 386.27 | 384.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 376.75 | 386.27 | 384.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 374.20 | 383.85 | 383.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 372.60 | 376.14 | 378.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 376.10 | 375.35 | 377.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 376.10 | 375.35 | 377.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 376.10 | 375.35 | 377.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 376.50 | 375.35 | 377.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 380.40 | 375.05 | 375.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 380.05 | 375.05 | 375.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 379.35 | 375.91 | 376.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:00:00 | 377.95 | 376.32 | 376.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 381.20 | 377.30 | 376.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 381.20 | 377.30 | 376.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 382.80 | 379.07 | 377.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 386.85 | 387.56 | 384.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 386.85 | 387.56 | 384.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 420.20 | 426.19 | 419.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 418.35 | 426.19 | 419.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 419.55 | 424.86 | 419.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 419.40 | 424.86 | 419.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 418.55 | 423.60 | 419.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 418.85 | 423.60 | 419.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 418.50 | 422.58 | 419.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 422.90 | 422.58 | 419.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:15:00 | 420.60 | 420.63 | 420.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 12:15:00 | 418.65 | 419.75 | 419.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 418.65 | 419.75 | 419.82 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 423.00 | 420.44 | 420.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 424.50 | 421.62 | 420.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 420.00 | 422.33 | 421.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 420.00 | 422.33 | 421.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 420.00 | 422.33 | 421.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 420.00 | 422.33 | 421.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 420.95 | 422.05 | 421.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:30:00 | 420.75 | 422.05 | 421.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 423.60 | 422.36 | 421.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 421.80 | 422.36 | 421.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 422.85 | 423.13 | 422.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:30:00 | 426.55 | 423.90 | 422.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 414.40 | 428.30 | 428.11 | SL hit (close<static) qty=1.00 sl=421.55 alert=retest2 |

### Cycle 118 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 420.15 | 426.67 | 427.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 381.20 | 415.28 | 421.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 399.00 | 398.07 | 407.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 399.00 | 398.07 | 407.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 399.00 | 398.07 | 407.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 397.20 | 398.07 | 407.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 393.55 | 400.86 | 405.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:15:00 | 397.70 | 401.28 | 403.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:00:00 | 397.65 | 400.55 | 402.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 411.30 | 402.49 | 403.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 411.30 | 402.49 | 403.37 | SL hit (close>static) qty=1.00 sl=410.90 alert=retest2 |

### Cycle 119 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 413.40 | 405.78 | 404.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 423.85 | 413.75 | 409.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 433.00 | 433.26 | 428.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 442.15 | 433.26 | 428.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 429.30 | 432.47 | 428.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 10:00:00 | 429.30 | 432.47 | 428.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 427.35 | 431.45 | 428.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-21 10:15:00 | 427.35 | 431.45 | 428.69 | SL hit (close<ema400) qty=1.00 sl=428.69 alert=retest1 |

### Cycle 120 — SELL (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 12:15:00 | 422.95 | 427.25 | 427.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 14:15:00 | 421.65 | 425.57 | 426.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 396.50 | 392.17 | 399.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 396.50 | 392.17 | 399.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 365.15 | 362.20 | 366.24 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 13:15:00 | 368.45 | 367.17 | 367.08 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 363.15 | 366.36 | 366.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 360.00 | 365.09 | 366.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 363.25 | 362.22 | 364.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 12:15:00 | 363.25 | 362.22 | 364.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 363.25 | 362.22 | 364.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 362.80 | 362.22 | 364.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 365.35 | 362.84 | 364.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 365.35 | 362.84 | 364.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 364.40 | 363.15 | 364.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:15:00 | 364.80 | 363.15 | 364.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 364.80 | 363.48 | 364.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 372.40 | 363.48 | 364.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 372.75 | 365.34 | 365.08 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 354.00 | 364.45 | 365.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 350.15 | 361.59 | 363.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 361.25 | 360.32 | 362.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 11:45:00 | 361.05 | 360.32 | 362.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 366.50 | 361.77 | 362.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 366.50 | 361.77 | 362.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 367.00 | 362.82 | 363.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 375.25 | 362.82 | 363.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 375.00 | 365.25 | 364.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 380.50 | 376.63 | 373.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 398.15 | 398.77 | 392.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 398.15 | 398.77 | 392.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 393.45 | 397.68 | 393.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 391.85 | 396.74 | 393.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 394.85 | 396.36 | 393.51 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 391.40 | 393.18 | 393.27 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 12:15:00 | 393.85 | 393.28 | 393.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 395.95 | 393.98 | 393.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 390.65 | 393.46 | 393.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 390.65 | 393.46 | 393.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 390.65 | 393.46 | 393.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:15:00 | 391.00 | 393.46 | 393.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 393.15 | 393.40 | 393.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 390.75 | 393.40 | 393.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 393.65 | 393.45 | 393.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 393.10 | 393.45 | 393.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 393.60 | 393.48 | 393.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 393.50 | 393.48 | 393.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 393.30 | 393.44 | 393.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 393.20 | 393.44 | 393.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 393.05 | 393.36 | 393.38 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 401.00 | 394.83 | 394.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 404.15 | 398.35 | 396.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 402.00 | 402.33 | 399.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:00:00 | 402.00 | 402.33 | 399.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 407.30 | 403.19 | 400.30 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 400.15 | 402.68 | 402.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 396.70 | 400.34 | 401.63 | Break + close below crossover candle low |

### Cycle 131 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 413.30 | 400.63 | 400.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 418.60 | 404.22 | 402.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 423.60 | 424.30 | 419.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:45:00 | 422.75 | 424.30 | 419.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 416.40 | 421.95 | 420.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 416.40 | 421.95 | 420.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 414.00 | 420.36 | 419.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 419.20 | 420.36 | 419.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:45:00 | 417.15 | 419.99 | 419.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 10:15:00 | 415.85 | 419.16 | 419.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 415.85 | 419.16 | 419.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 13:15:00 | 414.00 | 416.96 | 418.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 410.50 | 409.63 | 411.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 410.50 | 409.63 | 411.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 410.50 | 409.63 | 411.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 410.50 | 409.63 | 411.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 408.00 | 409.50 | 411.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 406.00 | 408.66 | 410.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:30:00 | 407.00 | 408.23 | 410.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 10:45:00 | 406.15 | 403.98 | 407.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:15:00 | 385.70 | 389.45 | 393.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:15:00 | 386.65 | 389.45 | 393.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:15:00 | 385.84 | 389.45 | 393.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 387.85 | 386.29 | 389.98 | SL hit (close>ema200) qty=0.50 sl=386.29 alert=retest2 |

### Cycle 133 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 395.20 | 389.67 | 389.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 408.40 | 396.75 | 393.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 417.70 | 418.31 | 412.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 417.70 | 418.31 | 412.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 415.80 | 417.24 | 415.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:15:00 | 416.00 | 417.24 | 415.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 415.55 | 416.90 | 415.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 415.15 | 416.90 | 415.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 415.15 | 416.55 | 415.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 415.15 | 416.55 | 415.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 414.70 | 416.18 | 415.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 414.70 | 416.18 | 415.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 415.20 | 415.98 | 415.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 415.20 | 415.98 | 415.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 413.55 | 415.50 | 415.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 417.75 | 415.50 | 415.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 411.85 | 414.77 | 414.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 411.85 | 414.77 | 414.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 410.40 | 413.89 | 414.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 414.25 | 413.97 | 414.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 414.25 | 413.97 | 414.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 414.25 | 413.97 | 414.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 414.25 | 413.97 | 414.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 412.95 | 413.76 | 414.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:30:00 | 413.70 | 413.76 | 414.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 414.95 | 413.82 | 414.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 414.95 | 413.82 | 414.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 414.35 | 413.93 | 414.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 413.10 | 413.93 | 414.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 413.10 | 413.58 | 414.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 412.15 | 413.58 | 414.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 411.30 | 412.15 | 413.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 405.60 | 409.04 | 410.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 403.95 | 408.36 | 410.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 420.50 | 407.54 | 406.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 420.50 | 407.54 | 406.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 426.95 | 411.42 | 408.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 428.10 | 429.55 | 423.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 428.10 | 429.55 | 423.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 423.65 | 428.37 | 423.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 423.65 | 428.37 | 423.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 425.25 | 427.75 | 423.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 427.60 | 427.75 | 423.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 418.25 | 423.79 | 423.06 | SL hit (close<static) qty=1.00 sl=422.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 417.10 | 422.45 | 422.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 416.10 | 421.18 | 421.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 418.90 | 417.95 | 419.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 418.90 | 417.95 | 419.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 418.90 | 417.95 | 419.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 419.25 | 417.95 | 419.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 419.45 | 418.25 | 419.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 419.95 | 418.25 | 419.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 423.00 | 419.20 | 420.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 423.00 | 419.20 | 420.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 427.95 | 420.95 | 420.75 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 421.70 | 422.76 | 422.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 420.00 | 421.59 | 422.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 424.40 | 421.52 | 421.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 424.40 | 421.52 | 421.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 424.40 | 421.52 | 421.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 425.95 | 421.52 | 421.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 426.45 | 422.51 | 422.34 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 420.10 | 422.20 | 422.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 418.65 | 420.72 | 421.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 388.65 | 386.04 | 390.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 388.65 | 386.04 | 390.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 388.65 | 386.04 | 390.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 389.30 | 386.04 | 390.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 391.00 | 387.66 | 390.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 391.00 | 387.66 | 390.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 392.55 | 388.64 | 390.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 392.55 | 388.64 | 390.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 390.70 | 389.05 | 390.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 390.50 | 389.05 | 390.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 392.60 | 389.76 | 390.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:15:00 | 395.05 | 389.76 | 390.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 395.75 | 390.96 | 391.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 395.30 | 390.96 | 391.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 394.15 | 391.60 | 391.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 397.40 | 393.10 | 392.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 393.85 | 396.47 | 394.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 393.85 | 396.47 | 394.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 393.85 | 396.47 | 394.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 393.85 | 396.47 | 394.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 392.50 | 395.67 | 394.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 396.00 | 395.67 | 394.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 395.50 | 395.65 | 394.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 392.50 | 397.01 | 397.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 392.50 | 397.01 | 397.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 389.50 | 395.51 | 396.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 383.80 | 383.47 | 387.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 383.80 | 383.47 | 387.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 379.55 | 373.61 | 376.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 379.55 | 373.61 | 376.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 383.55 | 375.59 | 376.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 382.10 | 375.59 | 376.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 385.00 | 378.92 | 378.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 15:15:00 | 387.40 | 380.61 | 379.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 381.45 | 382.41 | 380.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 381.45 | 382.41 | 380.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 381.45 | 382.41 | 380.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 380.65 | 382.41 | 380.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 379.80 | 381.73 | 380.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 379.80 | 381.73 | 380.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 379.40 | 381.26 | 380.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 381.15 | 381.26 | 380.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 377.50 | 380.28 | 380.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 377.50 | 380.28 | 380.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 376.00 | 379.42 | 379.81 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 391.60 | 381.29 | 380.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 393.40 | 385.56 | 382.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 391.25 | 392.54 | 389.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 389.80 | 392.54 | 389.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 392.35 | 392.50 | 389.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 392.00 | 392.50 | 389.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 389.40 | 392.11 | 390.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 389.40 | 392.11 | 390.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 389.55 | 391.60 | 390.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 393.05 | 391.26 | 390.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 384.50 | 389.34 | 389.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 384.50 | 389.34 | 389.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 380.05 | 385.80 | 387.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 383.50 | 382.67 | 385.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 383.50 | 382.67 | 385.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 383.50 | 382.67 | 385.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 380.65 | 382.55 | 384.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 376.35 | 382.28 | 383.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 380.50 | 381.13 | 382.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 381.10 | 381.00 | 382.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 376.15 | 375.68 | 377.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 377.00 | 375.68 | 377.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 378.40 | 376.23 | 377.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:45:00 | 377.25 | 376.23 | 377.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 370.55 | 375.10 | 377.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 379.10 | 375.10 | 377.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 378.00 | 375.17 | 376.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 377.60 | 375.17 | 376.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 376.35 | 375.40 | 376.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 375.75 | 375.40 | 376.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 379.90 | 375.79 | 376.13 | SL hit (close>static) qty=1.00 sl=378.10 alert=retest2 |

### Cycle 147 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 379.80 | 376.59 | 376.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 381.70 | 379.00 | 377.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 381.45 | 381.47 | 379.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 381.45 | 381.47 | 379.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 381.45 | 381.47 | 379.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 379.70 | 381.47 | 379.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 378.60 | 380.90 | 379.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 378.60 | 380.90 | 379.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 378.80 | 380.48 | 379.66 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 376.50 | 378.86 | 379.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 373.65 | 377.03 | 378.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 385.55 | 377.70 | 377.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 385.55 | 377.70 | 377.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 385.55 | 377.70 | 377.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 383.55 | 377.70 | 377.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 384.20 | 379.00 | 378.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 396.00 | 386.79 | 383.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 388.35 | 389.37 | 386.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 388.35 | 389.37 | 386.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 392.85 | 392.82 | 390.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 395.50 | 392.82 | 390.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 389.50 | 392.14 | 390.60 | SL hit (close<static) qty=1.00 sl=390.10 alert=retest2 |

### Cycle 150 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 384.35 | 388.71 | 389.27 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 392.85 | 389.84 | 389.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 394.25 | 391.98 | 391.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 12:15:00 | 391.75 | 391.94 | 391.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 12:15:00 | 391.75 | 391.94 | 391.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 391.75 | 391.94 | 391.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 391.75 | 391.94 | 391.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 392.20 | 391.99 | 391.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 392.20 | 391.99 | 391.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 391.20 | 391.93 | 391.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 390.15 | 391.93 | 391.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 390.35 | 391.61 | 391.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 390.10 | 391.61 | 391.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 389.80 | 391.25 | 391.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 390.10 | 391.25 | 391.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 389.00 | 390.80 | 390.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 15:15:00 | 388.20 | 389.50 | 390.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 12:15:00 | 389.40 | 388.28 | 389.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 12:15:00 | 389.40 | 388.28 | 389.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 389.40 | 388.28 | 389.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 389.40 | 388.28 | 389.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 394.55 | 389.54 | 389.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:30:00 | 393.25 | 389.54 | 389.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 413.10 | 394.25 | 391.89 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 391.80 | 396.91 | 397.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 391.25 | 394.45 | 395.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 378.80 | 378.59 | 383.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 378.80 | 378.59 | 383.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 375.60 | 374.42 | 376.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 376.60 | 374.42 | 376.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 375.80 | 374.69 | 376.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 376.00 | 374.69 | 376.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 377.25 | 375.20 | 376.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 377.25 | 375.20 | 376.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 380.70 | 376.30 | 377.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:45:00 | 380.40 | 376.30 | 377.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 382.35 | 377.51 | 377.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 382.45 | 377.51 | 377.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 382.65 | 378.54 | 377.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 384.50 | 380.93 | 379.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 382.30 | 383.01 | 381.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 383.60 | 383.01 | 381.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 379.65 | 382.34 | 381.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 379.60 | 382.34 | 381.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 379.35 | 381.74 | 380.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 379.70 | 381.74 | 380.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 377.80 | 380.05 | 380.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 376.60 | 378.73 | 379.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 379.20 | 378.60 | 379.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 379.20 | 378.60 | 379.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 379.50 | 378.80 | 379.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 380.00 | 378.80 | 379.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 379.95 | 379.03 | 379.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 380.60 | 379.03 | 379.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 382.50 | 379.72 | 379.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 12:15:00 | 385.50 | 381.35 | 380.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 386.00 | 387.07 | 384.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 386.00 | 387.07 | 384.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 386.00 | 387.43 | 386.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 383.55 | 387.43 | 386.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 382.30 | 386.40 | 385.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 383.05 | 386.40 | 385.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 382.40 | 385.60 | 385.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 380.85 | 383.02 | 384.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 381.30 | 379.62 | 380.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 12:15:00 | 381.30 | 379.62 | 380.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 381.30 | 379.62 | 380.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 381.30 | 379.62 | 380.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 378.35 | 379.37 | 380.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:00:00 | 375.90 | 378.67 | 380.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 376.85 | 378.07 | 378.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 13:15:00 | 357.10 | 372.70 | 375.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 13:15:00 | 358.01 | 372.70 | 375.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 369.15 | 364.37 | 367.71 | SL hit (close>ema200) qty=0.50 sl=364.37 alert=retest2 |

### Cycle 159 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 370.00 | 368.49 | 368.48 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 366.65 | 368.13 | 368.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 366.15 | 367.51 | 367.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 367.90 | 367.35 | 367.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 367.90 | 367.35 | 367.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 367.90 | 367.35 | 367.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 367.90 | 367.35 | 367.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 368.20 | 367.52 | 367.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 364.95 | 367.52 | 367.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 371.00 | 366.38 | 366.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 371.00 | 366.38 | 366.18 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 364.90 | 368.36 | 368.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 364.45 | 367.58 | 368.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 364.80 | 363.60 | 365.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 11:15:00 | 364.80 | 363.60 | 365.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 364.80 | 363.60 | 365.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 367.25 | 363.60 | 365.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 354.75 | 352.62 | 355.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:45:00 | 355.75 | 352.62 | 355.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 352.50 | 352.69 | 354.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 354.35 | 352.69 | 354.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 355.20 | 353.19 | 354.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 355.20 | 353.19 | 354.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 352.95 | 353.15 | 354.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 355.45 | 353.15 | 354.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 351.95 | 350.11 | 351.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 351.95 | 350.11 | 351.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 353.25 | 350.73 | 351.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 353.40 | 350.73 | 351.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 351.45 | 350.94 | 351.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 351.45 | 350.94 | 351.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 350.80 | 350.91 | 351.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 351.50 | 350.91 | 351.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 348.90 | 350.51 | 351.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 15:00:00 | 345.75 | 348.52 | 349.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 344.80 | 348.27 | 348.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 344.90 | 346.97 | 347.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 328.46 | 332.27 | 335.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 327.56 | 332.27 | 335.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 327.65 | 332.27 | 335.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 326.60 | 326.45 | 328.97 | SL hit (close>ema200) qty=0.50 sl=326.45 alert=retest2 |

### Cycle 163 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 321.05 | 313.18 | 313.06 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 309.35 | 312.72 | 313.08 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 315.00 | 313.49 | 313.31 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 312.30 | 313.10 | 313.20 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 314.90 | 313.46 | 313.35 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 312.40 | 313.13 | 313.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 12:15:00 | 312.05 | 312.92 | 313.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 314.10 | 313.10 | 313.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 314.10 | 313.10 | 313.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 314.10 | 313.10 | 313.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 314.10 | 313.10 | 313.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 313.55 | 313.19 | 313.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 310.35 | 313.19 | 313.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:00:00 | 312.95 | 313.00 | 313.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 314.20 | 313.24 | 313.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 314.20 | 313.24 | 313.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 316.35 | 313.96 | 313.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 313.65 | 314.68 | 314.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 313.65 | 314.68 | 314.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 313.65 | 314.68 | 314.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 313.00 | 314.68 | 314.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 313.60 | 314.47 | 314.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 313.60 | 314.47 | 314.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 313.30 | 314.23 | 313.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 312.60 | 314.23 | 313.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 312.15 | 313.60 | 313.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 311.15 | 313.11 | 313.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 312.75 | 312.08 | 312.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 11:15:00 | 312.75 | 312.08 | 312.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 312.75 | 312.08 | 312.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 312.75 | 312.08 | 312.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 312.15 | 312.09 | 312.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 312.30 | 312.09 | 312.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 313.10 | 312.29 | 312.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 313.10 | 312.29 | 312.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 310.10 | 311.85 | 312.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 308.40 | 310.78 | 311.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 313.45 | 311.19 | 311.36 | SL hit (close>static) qty=1.00 sl=313.35 alert=retest2 |

### Cycle 171 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 312.80 | 311.51 | 311.49 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 309.75 | 311.16 | 311.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 13:15:00 | 306.90 | 310.31 | 310.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 310.85 | 309.81 | 310.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 310.85 | 309.81 | 310.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 310.85 | 309.81 | 310.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 310.85 | 309.81 | 310.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 311.35 | 310.12 | 310.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 312.15 | 310.12 | 310.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 312.35 | 310.49 | 310.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 309.10 | 310.11 | 310.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:00:00 | 308.30 | 309.39 | 309.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 12:15:00 | 293.64 | 301.08 | 304.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 12:15:00 | 292.88 | 301.08 | 304.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 294.55 | 291.26 | 295.70 | SL hit (close>ema200) qty=0.50 sl=291.26 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 302.65 | 298.77 | 298.39 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 297.55 | 298.63 | 298.67 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 301.25 | 298.72 | 298.65 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 297.80 | 298.58 | 298.62 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 300.80 | 299.01 | 298.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 301.85 | 299.58 | 299.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 299.30 | 299.52 | 299.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 299.30 | 299.52 | 299.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 299.30 | 299.52 | 299.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 300.50 | 299.52 | 299.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 298.45 | 299.31 | 299.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 298.05 | 299.31 | 299.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 299.45 | 299.34 | 299.08 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 297.05 | 298.80 | 298.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 294.85 | 297.79 | 298.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 11:15:00 | 294.15 | 293.75 | 295.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 294.15 | 293.75 | 295.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 294.40 | 293.97 | 294.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:30:00 | 295.50 | 293.97 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 292.80 | 293.71 | 294.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 291.55 | 293.28 | 294.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 276.97 | 284.25 | 287.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 281.00 | 280.61 | 283.59 | SL hit (close>ema200) qty=0.50 sl=280.61 alert=retest2 |

### Cycle 179 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 275.95 | 270.90 | 270.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 277.90 | 272.30 | 271.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 273.50 | 273.59 | 271.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 10:00:00 | 273.50 | 273.59 | 271.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 269.90 | 272.85 | 271.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 269.90 | 272.85 | 271.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 265.95 | 271.47 | 271.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 265.95 | 271.47 | 271.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 266.70 | 270.52 | 270.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 265.20 | 268.75 | 269.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 265.45 | 263.03 | 265.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 265.45 | 263.03 | 265.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 265.45 | 263.03 | 265.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 265.45 | 263.03 | 265.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 265.70 | 263.56 | 265.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 265.70 | 263.56 | 265.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 269.50 | 264.75 | 265.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 269.50 | 264.75 | 265.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 271.55 | 266.11 | 266.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 271.55 | 266.11 | 266.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 270.80 | 267.05 | 266.66 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 265.95 | 266.70 | 266.75 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 267.30 | 266.70 | 266.69 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 266.40 | 266.64 | 266.67 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 268.30 | 266.97 | 266.81 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 15:15:00 | 263.70 | 266.23 | 266.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 262.25 | 264.87 | 265.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 263.20 | 262.98 | 264.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 13:15:00 | 264.75 | 263.34 | 264.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 264.75 | 263.34 | 264.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 264.75 | 263.34 | 264.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 267.70 | 264.21 | 264.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 267.70 | 264.21 | 264.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 267.15 | 264.80 | 264.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 297.90 | 264.80 | 264.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 302.80 | 272.40 | 268.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 316.10 | 281.14 | 272.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 280.85 | 292.96 | 284.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 280.85 | 292.96 | 284.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 280.85 | 292.96 | 284.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 280.10 | 292.96 | 284.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 279.75 | 290.32 | 283.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 279.75 | 290.32 | 283.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 282.20 | 287.54 | 283.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 281.80 | 287.54 | 283.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 282.85 | 285.83 | 283.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 282.15 | 285.83 | 283.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 281.75 | 284.56 | 283.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 280.75 | 284.56 | 283.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 279.50 | 283.55 | 282.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 279.50 | 283.55 | 282.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 282.00 | 282.86 | 282.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 282.00 | 282.86 | 282.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 281.90 | 282.67 | 282.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:15:00 | 280.20 | 282.67 | 282.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 280.20 | 282.17 | 282.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 271.10 | 279.96 | 281.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 275.05 | 273.79 | 276.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:45:00 | 275.55 | 273.79 | 276.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 286.00 | 276.23 | 277.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 286.00 | 276.23 | 277.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 286.25 | 278.23 | 278.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 290.55 | 278.23 | 278.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 299.90 | 282.57 | 280.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 324.35 | 300.23 | 293.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 315.40 | 315.56 | 306.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 303.60 | 312.61 | 309.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 303.60 | 312.61 | 309.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 303.60 | 312.61 | 309.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 303.05 | 310.70 | 308.83 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 302.15 | 306.99 | 307.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 301.60 | 305.91 | 306.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 301.50 | 301.05 | 303.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 316.15 | 301.05 | 303.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 191 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 321.05 | 305.05 | 304.83 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 307.60 | 310.67 | 310.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 307.00 | 309.93 | 310.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 317.40 | 309.25 | 309.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 317.40 | 309.25 | 309.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 317.40 | 309.25 | 309.41 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 317.80 | 310.96 | 310.17 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 304.65 | 308.92 | 309.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 301.65 | 304.68 | 306.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 11:15:00 | 303.40 | 302.69 | 304.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 303.40 | 302.69 | 304.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 303.40 | 302.69 | 304.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 303.40 | 302.69 | 304.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 304.00 | 302.95 | 304.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 304.00 | 302.95 | 304.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 303.40 | 303.04 | 304.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 303.40 | 303.04 | 304.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 309.95 | 304.42 | 304.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 309.95 | 304.42 | 304.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 15:15:00 | 308.60 | 305.26 | 305.01 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 300.15 | 304.24 | 304.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 286.15 | 295.17 | 299.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 284.40 | 282.89 | 287.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 284.40 | 282.89 | 287.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 284.95 | 283.59 | 287.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 281.70 | 283.89 | 286.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 267.61 | 280.59 | 284.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 15:15:00 | 272.30 | 272.20 | 277.58 | SL hit (close>ema200) qty=0.50 sl=272.20 alert=retest2 |

### Cycle 197 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 266.80 | 240.43 | 240.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 270.00 | 246.34 | 242.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 256.95 | 260.75 | 253.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:00:00 | 256.95 | 260.75 | 253.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 255.55 | 258.61 | 254.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 255.55 | 258.61 | 254.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 254.00 | 257.69 | 254.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:30:00 | 254.30 | 257.69 | 254.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 256.00 | 257.35 | 254.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 247.50 | 257.35 | 254.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 250.35 | 255.95 | 254.25 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 246.65 | 252.32 | 252.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 241.45 | 248.47 | 250.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 256.50 | 248.88 | 250.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 256.50 | 248.88 | 250.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 256.50 | 248.88 | 250.48 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 260.16 | 251.86 | 251.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 273.42 | 262.48 | 259.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 270.75 | 270.82 | 267.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 270.94 | 270.82 | 267.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 273.80 | 273.54 | 271.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 265.02 | 273.54 | 271.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 269.00 | 272.63 | 270.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 272.03 | 272.41 | 270.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 270.51 | 271.04 | 270.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 270.05 | 270.55 | 270.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 281.67 | 288.43 | 288.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 281.67 | 288.43 | 288.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 280.31 | 285.75 | 287.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 289.38 | 285.01 | 286.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 289.38 | 285.01 | 286.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 289.38 | 285.01 | 286.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 290.80 | 285.01 | 286.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 293.00 | 286.61 | 286.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 292.90 | 286.61 | 286.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 293.38 | 287.96 | 287.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 296.66 | 292.15 | 290.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 291.10 | 296.55 | 295.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 291.10 | 296.55 | 295.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 291.10 | 296.55 | 295.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 291.10 | 296.55 | 295.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 293.00 | 295.84 | 294.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 292.13 | 295.84 | 294.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 293.50 | 295.37 | 294.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 292.60 | 295.37 | 294.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 289.35 | 293.69 | 294.02 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 301.80 | 294.07 | 294.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 306.20 | 296.50 | 295.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 301.15 | 303.17 | 300.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 301.15 | 303.17 | 300.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 301.15 | 303.17 | 300.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 300.10 | 303.17 | 300.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 300.50 | 302.28 | 300.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 300.50 | 302.28 | 300.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 304.30 | 302.68 | 300.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 306.00 | 302.68 | 300.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:30:00 | 304.90 | 303.46 | 302.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 304.95 | 304.07 | 302.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 306.00 | 304.52 | 303.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 306.00 | 304.81 | 303.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 308.65 | 304.81 | 303.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 309.40 | 305.93 | 304.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 308.50 | 306.18 | 304.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-25 10:15:00 | 267.50 | 2024-05-02 09:15:00 | 269.90 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-04-25 11:15:00 | 267.85 | 2024-05-02 09:15:00 | 269.90 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2024-04-25 14:30:00 | 267.25 | 2024-05-02 09:15:00 | 269.90 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2024-04-26 09:15:00 | 269.75 | 2024-05-02 09:15:00 | 269.90 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-05-08 12:45:00 | 261.25 | 2024-05-10 09:15:00 | 248.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 13:30:00 | 260.40 | 2024-05-10 09:15:00 | 247.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 257.80 | 2024-05-10 09:15:00 | 244.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 12:45:00 | 261.25 | 2024-05-10 13:15:00 | 254.35 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2024-05-08 13:30:00 | 260.40 | 2024-05-10 13:15:00 | 254.35 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2024-05-09 09:15:00 | 257.80 | 2024-05-10 13:15:00 | 254.35 | STOP_HIT | 0.50 | 1.34% |
| SELL | retest2 | 2024-05-29 09:15:00 | 243.15 | 2024-06-03 10:15:00 | 230.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 243.15 | 2024-06-03 15:15:00 | 234.90 | STOP_HIT | 0.50 | 3.39% |
| BUY | retest2 | 2024-06-19 11:45:00 | 253.00 | 2024-06-25 09:15:00 | 278.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-02 12:30:00 | 253.00 | 2024-07-04 09:15:00 | 258.80 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-07-03 09:15:00 | 253.15 | 2024-07-04 09:15:00 | 258.80 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-07-03 10:45:00 | 254.20 | 2024-07-04 09:15:00 | 258.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-07-03 12:15:00 | 254.20 | 2024-07-04 09:15:00 | 258.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-07-09 13:00:00 | 268.40 | 2024-07-10 09:15:00 | 262.05 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-07-15 09:15:00 | 276.00 | 2024-07-19 10:15:00 | 276.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-07-15 10:00:00 | 282.90 | 2024-07-19 10:15:00 | 276.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-07-19 10:15:00 | 277.75 | 2024-07-19 10:15:00 | 276.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-08-07 09:15:00 | 395.80 | 2024-08-12 13:15:00 | 391.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-08 09:30:00 | 392.00 | 2024-08-12 13:15:00 | 391.80 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-08-08 10:00:00 | 393.65 | 2024-08-12 13:15:00 | 391.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-08-08 11:45:00 | 395.60 | 2024-08-12 13:15:00 | 391.80 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-08-12 12:30:00 | 390.60 | 2024-08-12 13:15:00 | 391.80 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-08-22 14:15:00 | 442.95 | 2024-08-23 13:15:00 | 487.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 15:15:00 | 517.00 | 2024-09-27 13:15:00 | 568.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 09:30:00 | 523.15 | 2024-09-27 13:15:00 | 575.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 14:45:00 | 519.95 | 2024-09-27 13:15:00 | 571.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-04 14:45:00 | 542.95 | 2024-10-07 09:15:00 | 515.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 09:30:00 | 525.60 | 2024-10-07 14:15:00 | 499.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 14:45:00 | 542.95 | 2024-10-08 09:15:00 | 488.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-07 09:30:00 | 525.60 | 2024-10-09 09:15:00 | 516.65 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-10-17 09:45:00 | 493.35 | 2024-10-18 09:15:00 | 468.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:45:00 | 493.35 | 2024-10-18 10:15:00 | 486.90 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2024-11-04 09:15:00 | 412.25 | 2024-11-06 09:15:00 | 429.80 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2024-11-12 12:15:00 | 414.00 | 2024-11-13 09:15:00 | 393.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 414.00 | 2024-11-14 09:15:00 | 400.40 | STOP_HIT | 0.50 | 3.29% |
| BUY | retest2 | 2024-11-26 09:15:00 | 398.90 | 2024-11-27 12:15:00 | 435.55 | TARGET_HIT | 1.00 | 9.19% |
| BUY | retest2 | 2024-11-26 10:00:00 | 395.95 | 2024-11-27 12:15:00 | 434.72 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2024-11-26 12:00:00 | 395.20 | 2024-11-27 12:15:00 | 434.78 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2024-11-26 13:00:00 | 395.25 | 2024-11-28 09:15:00 | 438.79 | TARGET_HIT | 1.00 | 11.02% |
| BUY | retest2 | 2024-12-18 11:15:00 | 491.10 | 2024-12-19 09:15:00 | 473.60 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-12-18 13:00:00 | 489.15 | 2024-12-19 09:15:00 | 473.60 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-12-26 14:45:00 | 458.65 | 2024-12-30 13:15:00 | 435.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 15:15:00 | 457.90 | 2024-12-30 13:15:00 | 435.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 10:15:00 | 457.85 | 2024-12-30 13:15:00 | 434.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 14:00:00 | 457.45 | 2024-12-30 13:15:00 | 434.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 09:15:00 | 458.75 | 2024-12-30 13:15:00 | 435.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 11:00:00 | 460.15 | 2024-12-30 13:15:00 | 437.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 14:45:00 | 458.65 | 2024-12-31 11:15:00 | 446.75 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2024-12-26 15:15:00 | 457.90 | 2024-12-31 11:15:00 | 446.75 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2024-12-27 10:15:00 | 457.85 | 2024-12-31 11:15:00 | 446.75 | STOP_HIT | 0.50 | 2.42% |
| SELL | retest2 | 2024-12-27 14:00:00 | 457.45 | 2024-12-31 11:15:00 | 446.75 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2024-12-30 09:15:00 | 458.75 | 2024-12-31 11:15:00 | 446.75 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2024-12-30 11:00:00 | 460.15 | 2024-12-31 11:15:00 | 446.75 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-01-07 12:45:00 | 440.90 | 2025-01-08 14:15:00 | 418.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 12:45:00 | 440.90 | 2025-01-09 14:15:00 | 396.81 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 09:15:00 | 367.05 | 2025-02-03 14:15:00 | 403.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-14 09:45:00 | 372.85 | 2025-02-14 13:15:00 | 354.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:45:00 | 372.85 | 2025-02-17 09:15:00 | 370.25 | STOP_HIT | 0.50 | 0.70% |
| SELL | retest2 | 2025-02-17 10:00:00 | 370.25 | 2025-02-18 13:15:00 | 373.85 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-02-17 10:45:00 | 372.40 | 2025-02-18 13:15:00 | 373.85 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-02-18 11:00:00 | 372.25 | 2025-02-18 13:15:00 | 373.85 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-20 12:00:00 | 387.85 | 2025-02-24 09:15:00 | 382.85 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest1 | 2025-02-20 12:30:00 | 387.85 | 2025-02-24 09:15:00 | 382.85 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest1 | 2025-02-20 13:45:00 | 389.35 | 2025-02-24 09:15:00 | 382.85 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-03-18 12:00:00 | 377.95 | 2025-03-18 12:15:00 | 381.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-03-26 09:15:00 | 422.90 | 2025-03-27 12:15:00 | 418.65 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-03-27 10:15:00 | 420.60 | 2025-03-27 12:15:00 | 418.65 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-04-02 10:30:00 | 426.55 | 2025-04-04 09:15:00 | 414.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-04-08 10:15:00 | 397.20 | 2025-04-11 09:15:00 | 411.30 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-04-09 09:15:00 | 393.55 | 2025-04-11 09:15:00 | 411.30 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-04-09 14:15:00 | 397.70 | 2025-04-11 09:15:00 | 411.30 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-04-09 15:00:00 | 397.65 | 2025-04-11 09:15:00 | 411.30 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest1 | 2025-04-21 09:15:00 | 442.15 | 2025-04-21 10:15:00 | 427.35 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-06-09 09:15:00 | 419.20 | 2025-06-09 10:15:00 | 415.85 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-09 09:45:00 | 417.15 | 2025-06-09 10:15:00 | 415.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-12 12:00:00 | 406.00 | 2025-06-19 09:15:00 | 385.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 12:30:00 | 407.00 | 2025-06-19 09:15:00 | 386.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 10:45:00 | 406.15 | 2025-06-19 09:15:00 | 385.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 12:00:00 | 406.00 | 2025-06-19 14:15:00 | 387.85 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-06-12 12:30:00 | 407.00 | 2025-06-19 14:15:00 | 387.85 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-06-13 10:45:00 | 406.15 | 2025-06-19 14:15:00 | 387.85 | STOP_HIT | 0.50 | 4.51% |
| BUY | retest2 | 2025-07-02 09:15:00 | 417.75 | 2025-07-02 09:15:00 | 411.85 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-03 09:15:00 | 413.10 | 2025-07-09 10:15:00 | 420.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-07-03 11:30:00 | 413.10 | 2025-07-09 10:15:00 | 420.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-07-03 12:00:00 | 412.15 | 2025-07-09 10:15:00 | 420.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-07-04 09:30:00 | 411.30 | 2025-07-09 10:15:00 | 420.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-07-07 11:15:00 | 403.95 | 2025-07-09 10:15:00 | 420.50 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-07-11 12:15:00 | 427.60 | 2025-07-14 09:15:00 | 418.25 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-08-01 09:15:00 | 396.00 | 2025-08-06 09:15:00 | 392.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-01 11:45:00 | 395.50 | 2025-08-06 09:15:00 | 392.50 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-08-21 09:15:00 | 393.05 | 2025-08-21 13:15:00 | 384.50 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-25 15:00:00 | 380.65 | 2025-09-02 09:15:00 | 379.90 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-26 09:15:00 | 376.35 | 2025-09-02 10:15:00 | 379.80 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-08-26 11:30:00 | 380.50 | 2025-09-02 10:15:00 | 379.80 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-26 14:15:00 | 381.10 | 2025-09-02 10:15:00 | 379.80 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-09-01 11:15:00 | 375.75 | 2025-09-02 10:15:00 | 379.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-12 09:15:00 | 395.50 | 2025-09-12 11:15:00 | 389.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-10-15 15:00:00 | 375.90 | 2025-10-17 13:15:00 | 357.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 09:15:00 | 376.85 | 2025-10-17 13:15:00 | 358.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 15:00:00 | 375.90 | 2025-10-21 13:15:00 | 369.15 | STOP_HIT | 0.50 | 1.80% |
| SELL | retest2 | 2025-10-17 09:15:00 | 376.85 | 2025-10-21 13:15:00 | 369.15 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2025-10-27 09:15:00 | 364.95 | 2025-10-29 10:15:00 | 371.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-11-13 15:00:00 | 345.75 | 2025-11-24 09:15:00 | 328.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 12:15:00 | 344.80 | 2025-11-24 09:15:00 | 327.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 344.90 | 2025-11-24 09:15:00 | 327.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 15:00:00 | 345.75 | 2025-11-25 14:15:00 | 326.60 | STOP_HIT | 0.50 | 5.54% |
| SELL | retest2 | 2025-11-14 12:15:00 | 344.80 | 2025-11-25 14:15:00 | 326.60 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest2 | 2025-11-18 09:15:00 | 344.90 | 2025-11-25 14:15:00 | 326.60 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2025-12-15 09:15:00 | 310.35 | 2025-12-15 11:15:00 | 314.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-12-15 11:00:00 | 312.95 | 2025-12-15 11:15:00 | 314.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-18 09:30:00 | 308.40 | 2025-12-19 10:15:00 | 313.45 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-12-23 13:45:00 | 309.10 | 2025-12-29 12:15:00 | 293.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:00:00 | 308.30 | 2025-12-29 12:15:00 | 292.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 13:45:00 | 309.10 | 2025-12-31 09:15:00 | 294.55 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-12-24 11:00:00 | 308.30 | 2025-12-31 09:15:00 | 294.55 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2026-01-08 11:00:00 | 291.55 | 2026-01-12 09:15:00 | 276.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 291.55 | 2026-01-12 15:15:00 | 281.00 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2026-03-06 14:45:00 | 281.70 | 2026-03-09 09:15:00 | 267.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 281.70 | 2026-03-09 15:15:00 | 272.30 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2026-04-13 10:45:00 | 272.03 | 2026-04-24 10:15:00 | 281.67 | STOP_HIT | 1.00 | 3.54% |
| BUY | retest2 | 2026-04-13 13:45:00 | 270.51 | 2026-04-24 10:15:00 | 281.67 | STOP_HIT | 1.00 | 4.13% |
| BUY | retest2 | 2026-04-13 15:15:00 | 270.05 | 2026-04-24 10:15:00 | 281.67 | STOP_HIT | 1.00 | 4.30% |
