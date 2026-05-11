# JK Tyre & Industries Ltd. (JKTYRE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 406.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 217 |
| ALERT1 | 146 |
| ALERT2 | 145 |
| ALERT2_SKIP | 91 |
| ALERT3 | 308 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 126 |
| PARTIAL | 19 |
| TARGET_HIT | 9 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 94
- **Target hits / Stop hits / Partials:** 9 / 123 / 19
- **Avg / median % per leg:** 0.69% / -0.75%
- **Sum % (uncompounded):** 104.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 14 | 25.9% | 5 | 48 | 1 | -0.03% | -1.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 0.82% | 3.3% |
| BUY @ 3rd Alert (retest2) | 50 | 11 | 22.0% | 5 | 45 | 0 | -0.10% | -5.1% |
| SELL (all) | 97 | 43 | 44.3% | 4 | 75 | 18 | 1.10% | 106.4% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 3 | 1 | 2.59% | 10.4% |
| SELL @ 3rd Alert (retest2) | 93 | 39 | 41.9% | 4 | 72 | 17 | 1.03% | 96.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 0 | 6 | 2 | 1.70% | 13.6% |
| retest2 (combined) | 143 | 50 | 35.0% | 9 | 117 | 17 | 0.64% | 90.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 12:15:00 | 196.25 | 198.17 | 198.43 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 11:15:00 | 200.60 | 198.57 | 198.40 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 197.05 | 198.48 | 198.52 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 15:15:00 | 199.95 | 198.75 | 198.62 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 09:15:00 | 188.30 | 196.66 | 197.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 11:15:00 | 187.00 | 193.39 | 195.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 12:15:00 | 178.90 | 178.05 | 182.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 11:15:00 | 179.75 | 178.98 | 180.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 179.75 | 178.98 | 180.93 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 13:15:00 | 182.45 | 181.31 | 181.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 09:15:00 | 184.80 | 182.23 | 181.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 14:15:00 | 181.90 | 183.36 | 182.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 181.90 | 183.36 | 182.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 181.90 | 183.36 | 182.59 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 10:15:00 | 181.95 | 182.81 | 182.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 12:15:00 | 181.70 | 182.52 | 182.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 09:15:00 | 185.00 | 182.50 | 182.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 185.00 | 182.50 | 182.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 185.00 | 182.50 | 182.58 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 10:15:00 | 184.40 | 182.88 | 182.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 188.70 | 184.80 | 183.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 185.80 | 186.75 | 185.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 185.80 | 186.75 | 185.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 185.80 | 186.75 | 185.59 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 14:15:00 | 184.95 | 185.58 | 185.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 183.75 | 185.13 | 185.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 185.50 | 184.72 | 185.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 185.50 | 184.72 | 185.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 185.50 | 184.72 | 185.01 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 09:15:00 | 186.80 | 185.36 | 185.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 11:15:00 | 189.05 | 187.21 | 186.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 15:15:00 | 186.95 | 187.37 | 186.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 15:15:00 | 186.95 | 187.37 | 186.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 15:15:00 | 186.95 | 187.37 | 186.76 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 189.30 | 190.37 | 190.47 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 194.80 | 190.96 | 190.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 196.60 | 194.84 | 193.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 194.95 | 195.23 | 193.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 193.75 | 194.96 | 194.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 193.75 | 194.96 | 194.12 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 243.40 | 249.68 | 250.15 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 15:15:00 | 251.95 | 249.93 | 249.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 253.75 | 250.70 | 250.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 10:15:00 | 249.70 | 250.50 | 250.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 10:15:00 | 249.70 | 250.50 | 250.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 249.70 | 250.50 | 250.03 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 12:15:00 | 247.85 | 249.65 | 249.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 14:15:00 | 247.30 | 248.83 | 249.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 09:15:00 | 252.90 | 249.43 | 249.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 252.90 | 249.43 | 249.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 252.90 | 249.43 | 249.48 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 10:15:00 | 249.95 | 249.54 | 249.53 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 247.85 | 249.20 | 249.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 14:15:00 | 246.15 | 248.17 | 248.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 248.75 | 245.93 | 246.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 248.75 | 245.93 | 246.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 248.75 | 245.93 | 246.79 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 14:15:00 | 248.20 | 247.21 | 247.19 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 244.85 | 246.91 | 247.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 243.50 | 245.55 | 246.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 251.50 | 245.05 | 245.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 251.50 | 245.05 | 245.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 251.50 | 245.05 | 245.15 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 250.85 | 246.21 | 245.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 09:15:00 | 254.35 | 250.32 | 249.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 11:15:00 | 263.05 | 263.62 | 260.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 260.75 | 262.86 | 261.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 260.75 | 262.86 | 261.31 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 255.60 | 259.67 | 260.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 251.35 | 258.01 | 259.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 13:15:00 | 255.90 | 255.32 | 256.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 14:15:00 | 265.00 | 257.26 | 257.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 265.00 | 257.26 | 257.70 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 15:15:00 | 265.25 | 258.86 | 258.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 268.55 | 260.80 | 259.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 274.20 | 274.41 | 268.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 14:15:00 | 270.00 | 273.76 | 270.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 270.00 | 273.76 | 270.27 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 267.15 | 269.37 | 269.42 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 270.90 | 269.51 | 269.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 09:15:00 | 274.15 | 270.44 | 269.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 271.55 | 271.94 | 270.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 13:15:00 | 271.55 | 271.94 | 270.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 271.55 | 271.94 | 270.83 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 269.60 | 270.45 | 270.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 264.40 | 269.01 | 269.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 266.00 | 265.75 | 267.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 15:15:00 | 264.50 | 265.50 | 267.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 264.50 | 265.50 | 267.38 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 14:15:00 | 273.40 | 268.84 | 268.33 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 13:15:00 | 267.50 | 269.99 | 270.01 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 272.65 | 270.35 | 270.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 12:15:00 | 274.55 | 271.89 | 271.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 13:15:00 | 271.15 | 271.74 | 271.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 13:15:00 | 271.15 | 271.74 | 271.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 13:15:00 | 271.15 | 271.74 | 271.15 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 267.55 | 270.67 | 270.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 266.55 | 268.38 | 269.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 265.40 | 263.75 | 265.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 265.40 | 263.75 | 265.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 265.40 | 263.75 | 265.08 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 268.50 | 265.28 | 265.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 271.40 | 269.10 | 267.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 12:15:00 | 270.10 | 270.13 | 268.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 14:15:00 | 269.00 | 269.95 | 269.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 269.00 | 269.95 | 269.06 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 266.40 | 268.26 | 268.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 15:15:00 | 265.00 | 266.97 | 267.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 270.50 | 267.68 | 267.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 270.50 | 267.68 | 267.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 270.50 | 267.68 | 267.99 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 269.05 | 268.19 | 268.18 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 12:15:00 | 267.50 | 268.05 | 268.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 09:15:00 | 264.00 | 266.63 | 267.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 09:15:00 | 265.60 | 265.12 | 266.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 265.60 | 265.12 | 266.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 265.60 | 265.12 | 266.04 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 259.95 | 256.76 | 256.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 260.75 | 257.91 | 256.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 258.25 | 261.38 | 259.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 258.25 | 261.38 | 259.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 258.25 | 261.38 | 259.71 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 256.45 | 258.80 | 258.90 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 266.90 | 259.24 | 258.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-20 12:15:00 | 273.10 | 265.38 | 262.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 11:15:00 | 268.50 | 269.14 | 265.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 14:15:00 | 265.90 | 267.95 | 266.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 265.90 | 267.95 | 266.04 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 276.70 | 278.65 | 278.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 276.05 | 278.13 | 278.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 12:15:00 | 280.00 | 276.90 | 277.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 12:15:00 | 280.00 | 276.90 | 277.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 280.00 | 276.90 | 277.44 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 15:15:00 | 279.80 | 277.33 | 277.32 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 277.10 | 277.29 | 277.30 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 10:15:00 | 279.00 | 277.63 | 277.45 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 275.05 | 277.12 | 277.25 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 284.90 | 278.08 | 277.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 10:15:00 | 288.50 | 280.16 | 278.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 10:15:00 | 284.00 | 285.10 | 282.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 11:15:00 | 282.45 | 284.57 | 282.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 282.45 | 284.57 | 282.57 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 273.80 | 281.88 | 282.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 266.70 | 272.85 | 276.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 14:15:00 | 270.20 | 270.08 | 273.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 286.50 | 273.38 | 274.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 286.50 | 273.38 | 274.15 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 286.25 | 275.96 | 275.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 293.35 | 279.44 | 276.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 13:15:00 | 309.55 | 310.16 | 303.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 15:15:00 | 317.50 | 318.34 | 313.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 317.50 | 318.34 | 313.91 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 309.45 | 317.76 | 318.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 306.70 | 314.00 | 316.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 290.95 | 290.23 | 297.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 293.85 | 290.95 | 296.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 293.85 | 290.95 | 296.73 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 300.75 | 298.80 | 298.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 09:15:00 | 306.55 | 303.23 | 301.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 10:15:00 | 341.20 | 342.31 | 334.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 346.05 | 342.50 | 337.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 346.05 | 342.50 | 337.64 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 13:15:00 | 349.25 | 352.31 | 352.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 14:15:00 | 348.35 | 351.52 | 352.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 354.40 | 351.77 | 352.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 354.40 | 351.77 | 352.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 354.40 | 351.77 | 352.11 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 11:15:00 | 354.70 | 352.69 | 352.49 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 15:15:00 | 349.95 | 352.28 | 352.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 10:15:00 | 347.50 | 349.26 | 350.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 338.55 | 335.85 | 338.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 338.55 | 335.85 | 338.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 338.55 | 335.85 | 338.59 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 344.70 | 340.13 | 339.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 349.20 | 342.50 | 341.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 345.00 | 345.24 | 343.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 340.60 | 344.34 | 343.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 340.60 | 344.34 | 343.22 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 353.70 | 355.29 | 355.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 350.55 | 354.34 | 355.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 350.85 | 348.29 | 350.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 350.85 | 348.29 | 350.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 350.85 | 348.29 | 350.29 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 351.90 | 348.31 | 348.21 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 11:15:00 | 347.10 | 348.10 | 348.14 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 14:15:00 | 352.20 | 348.73 | 348.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 365.90 | 352.61 | 350.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 378.65 | 378.75 | 371.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 366.40 | 378.29 | 374.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 366.40 | 378.29 | 374.56 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 10:15:00 | 375.45 | 378.18 | 378.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 11:15:00 | 374.65 | 377.47 | 378.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 14:15:00 | 376.40 | 376.38 | 377.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 14:15:00 | 376.40 | 376.38 | 377.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 376.40 | 376.38 | 377.31 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 10:15:00 | 385.45 | 378.55 | 378.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 10:15:00 | 398.30 | 386.34 | 382.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 394.15 | 397.02 | 392.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 15:15:00 | 392.00 | 396.02 | 392.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 392.00 | 396.02 | 392.82 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 12:15:00 | 396.30 | 397.70 | 397.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 389.90 | 396.14 | 397.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 09:15:00 | 396.60 | 395.36 | 396.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 396.60 | 395.36 | 396.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 396.60 | 395.36 | 396.42 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 13:15:00 | 398.65 | 395.76 | 395.65 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 390.95 | 395.33 | 395.54 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 395.95 | 393.82 | 393.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 13:15:00 | 400.20 | 396.29 | 395.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 401.20 | 404.97 | 402.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 401.20 | 404.97 | 402.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 401.20 | 404.97 | 402.45 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 13:15:00 | 517.75 | 529.92 | 530.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 09:15:00 | 509.15 | 522.82 | 526.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 14:15:00 | 512.00 | 511.01 | 515.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 491.20 | 507.06 | 512.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 491.20 | 507.06 | 512.70 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 500.70 | 489.64 | 488.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 10:15:00 | 512.45 | 494.20 | 490.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 14:15:00 | 498.05 | 501.63 | 496.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 15:15:00 | 501.70 | 501.64 | 496.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 501.70 | 501.64 | 496.54 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 499.80 | 507.72 | 508.70 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 511.65 | 507.51 | 507.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 513.10 | 508.63 | 507.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 515.35 | 515.56 | 513.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 15:15:00 | 515.00 | 515.60 | 513.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 515.00 | 515.60 | 513.68 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 503.10 | 512.16 | 512.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 491.80 | 508.09 | 510.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 13:15:00 | 505.00 | 501.12 | 504.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 13:15:00 | 505.00 | 501.12 | 504.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 505.00 | 501.12 | 504.07 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 519.25 | 508.30 | 507.04 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 503.90 | 517.44 | 517.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 11:15:00 | 499.70 | 507.45 | 511.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 13:15:00 | 478.40 | 477.17 | 490.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 14:15:00 | 441.50 | 434.88 | 441.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 441.50 | 434.88 | 441.36 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 14:15:00 | 421.40 | 417.31 | 416.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 428.30 | 420.10 | 418.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 15:15:00 | 422.50 | 424.45 | 421.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 15:15:00 | 422.50 | 424.45 | 421.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 422.50 | 424.45 | 421.79 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 10:15:00 | 426.95 | 430.23 | 430.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 11:15:00 | 423.60 | 428.91 | 429.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 13:15:00 | 425.80 | 424.64 | 426.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 14:15:00 | 427.75 | 425.26 | 426.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 427.75 | 425.26 | 426.54 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 13:15:00 | 425.80 | 423.82 | 423.67 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 10:15:00 | 422.45 | 423.38 | 423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 417.70 | 421.91 | 422.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 411.00 | 409.72 | 413.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:00:00 | 411.00 | 409.72 | 413.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 409.10 | 409.55 | 411.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:45:00 | 407.65 | 409.44 | 411.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 418.35 | 407.03 | 407.73 | SL hit (close>static) qty=1.00 sl=413.95 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 422.10 | 410.04 | 409.04 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 13:15:00 | 406.75 | 409.89 | 410.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 14:15:00 | 404.55 | 408.82 | 409.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 409.20 | 408.08 | 409.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 409.20 | 408.08 | 409.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 409.20 | 408.08 | 409.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 409.85 | 408.08 | 409.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 408.90 | 408.25 | 409.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 11:30:00 | 407.50 | 408.10 | 409.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 12:30:00 | 407.40 | 408.03 | 408.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 13:45:00 | 407.25 | 407.86 | 408.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 09:15:00 | 412.00 | 408.40 | 408.73 | SL hit (close>static) qty=1.00 sl=409.45 alert=retest2 |

### Cycle 74 — BUY (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 10:15:00 | 413.35 | 409.39 | 409.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 415.90 | 413.40 | 411.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 12:15:00 | 413.35 | 413.82 | 412.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 13:00:00 | 413.35 | 413.82 | 412.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 411.80 | 413.42 | 412.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:00:00 | 411.80 | 413.42 | 412.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 408.55 | 412.44 | 411.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 408.55 | 412.44 | 411.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 410.65 | 411.73 | 411.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:30:00 | 410.70 | 411.73 | 411.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 11:15:00 | 410.55 | 411.49 | 411.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 13:15:00 | 407.75 | 410.57 | 411.10 | Break + close below crossover candle low |

### Cycle 76 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 417.50 | 411.31 | 411.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 420.25 | 413.09 | 412.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 15:15:00 | 425.00 | 426.50 | 422.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 09:15:00 | 418.10 | 426.50 | 422.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 418.30 | 424.86 | 422.02 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 415.95 | 419.79 | 420.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 409.50 | 417.39 | 418.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 12:15:00 | 404.10 | 399.13 | 403.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 12:15:00 | 404.10 | 399.13 | 403.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 404.10 | 399.13 | 403.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 404.10 | 399.13 | 403.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 401.60 | 399.63 | 403.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 399.55 | 399.63 | 403.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 398.00 | 400.68 | 403.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 399.70 | 401.03 | 403.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:00:00 | 400.15 | 400.57 | 402.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 379.57 | 395.21 | 398.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 379.71 | 395.21 | 398.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 380.14 | 395.21 | 398.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 378.10 | 386.22 | 391.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 381.90 | 381.87 | 387.74 | SL hit (close>ema200) qty=0.50 sl=381.87 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 392.75 | 387.00 | 386.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 401.10 | 391.33 | 388.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 11:15:00 | 401.15 | 401.51 | 398.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 12:00:00 | 401.15 | 401.51 | 398.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 416.45 | 417.69 | 416.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 416.45 | 417.69 | 416.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 416.00 | 417.35 | 416.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 424.00 | 417.35 | 416.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 414.25 | 419.61 | 418.52 | SL hit (close<static) qty=1.00 sl=415.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 414.40 | 417.77 | 417.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 411.60 | 415.75 | 416.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 406.35 | 403.81 | 406.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 406.35 | 403.81 | 406.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 406.40 | 404.32 | 406.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 406.40 | 404.32 | 406.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 404.00 | 404.26 | 406.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 410.70 | 404.26 | 406.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 411.00 | 405.61 | 407.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:00:00 | 408.80 | 406.25 | 407.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 411.00 | 408.01 | 407.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 411.00 | 408.01 | 407.86 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 392.45 | 404.99 | 406.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 370.80 | 398.15 | 403.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 390.75 | 385.74 | 392.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 390.75 | 385.74 | 392.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 391.30 | 386.85 | 392.44 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 410.80 | 395.41 | 394.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 413.70 | 408.21 | 404.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 409.45 | 409.71 | 406.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 409.15 | 409.71 | 406.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 406.30 | 408.99 | 406.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:00:00 | 406.30 | 408.99 | 406.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 405.50 | 408.29 | 406.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 405.50 | 408.29 | 406.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 407.85 | 408.21 | 406.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:15:00 | 408.85 | 408.21 | 406.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:15:00 | 408.50 | 408.06 | 406.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 410.40 | 407.87 | 407.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 404.50 | 406.94 | 406.83 | SL hit (close<static) qty=1.00 sl=404.75 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 405.70 | 406.69 | 406.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 10:15:00 | 403.10 | 405.16 | 405.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 12:15:00 | 400.50 | 400.02 | 401.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 13:00:00 | 400.50 | 400.02 | 401.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 401.80 | 400.34 | 401.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:00:00 | 401.80 | 400.34 | 401.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 401.60 | 400.59 | 401.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 397.15 | 400.59 | 401.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 393.75 | 399.23 | 400.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 11:15:00 | 393.40 | 398.33 | 400.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 12:30:00 | 393.65 | 396.98 | 399.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:00:00 | 393.40 | 395.94 | 398.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 09:45:00 | 393.45 | 395.09 | 397.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 393.50 | 394.34 | 395.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:30:00 | 391.25 | 394.95 | 395.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 392.05 | 392.90 | 393.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:00:00 | 391.20 | 392.56 | 393.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:45:00 | 391.60 | 392.49 | 393.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 399.60 | 393.91 | 394.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-25 14:15:00 | 399.60 | 393.91 | 394.05 | SL hit (close>static) qty=1.00 sl=396.85 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 403.40 | 395.81 | 394.90 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 15:15:00 | 393.95 | 395.44 | 395.63 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 416.95 | 399.74 | 397.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 10:15:00 | 418.60 | 403.51 | 399.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 12:15:00 | 459.25 | 459.80 | 455.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:00:00 | 459.25 | 459.80 | 455.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 456.70 | 459.18 | 455.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 456.65 | 459.18 | 455.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 457.50 | 458.84 | 455.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:30:00 | 455.00 | 458.84 | 455.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 457.00 | 458.47 | 455.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 454.70 | 458.47 | 455.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 454.50 | 457.68 | 455.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 461.65 | 457.50 | 455.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 11:15:00 | 459.60 | 457.50 | 455.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 445.30 | 453.84 | 454.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 445.30 | 453.84 | 454.61 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 455.40 | 452.33 | 452.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 465.15 | 457.34 | 454.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 09:15:00 | 456.80 | 458.84 | 456.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 456.80 | 458.84 | 456.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 456.80 | 458.84 | 456.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 459.60 | 458.84 | 456.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 456.80 | 458.43 | 456.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 457.85 | 458.43 | 456.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 454.95 | 457.74 | 456.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 455.10 | 457.74 | 456.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 454.15 | 457.02 | 456.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 453.75 | 457.02 | 456.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 450.60 | 455.19 | 455.54 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 492.10 | 461.98 | 458.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 500.40 | 469.66 | 462.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 476.55 | 478.43 | 471.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 10:00:00 | 476.55 | 478.43 | 471.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 473.40 | 476.63 | 472.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 472.60 | 476.63 | 472.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 474.50 | 476.21 | 473.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 469.10 | 476.21 | 473.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 476.00 | 476.16 | 473.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 467.00 | 476.16 | 473.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 463.00 | 473.53 | 472.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 463.00 | 473.53 | 472.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 465.00 | 471.83 | 471.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 463.85 | 471.83 | 471.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 464.70 | 470.40 | 471.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 460.80 | 467.46 | 469.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 440.55 | 435.28 | 440.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 440.55 | 435.28 | 440.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 440.55 | 435.28 | 440.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 440.55 | 435.28 | 440.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 439.65 | 436.15 | 440.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:45:00 | 435.65 | 440.08 | 441.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 11:15:00 | 437.85 | 439.72 | 440.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 13:15:00 | 437.85 | 439.08 | 440.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 437.05 | 438.96 | 439.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 436.20 | 438.41 | 439.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-26 13:15:00 | 446.10 | 440.16 | 440.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 446.10 | 440.16 | 440.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 452.00 | 444.95 | 443.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 449.00 | 450.40 | 446.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 449.00 | 450.40 | 446.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 447.60 | 449.84 | 447.05 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 444.00 | 446.39 | 446.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 442.35 | 445.49 | 445.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 410.45 | 410.25 | 416.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 411.15 | 410.25 | 416.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 416.70 | 412.15 | 415.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 416.70 | 412.15 | 415.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 416.70 | 413.06 | 415.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 416.40 | 413.06 | 415.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 412.70 | 412.99 | 415.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:30:00 | 408.70 | 411.06 | 413.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 09:30:00 | 407.85 | 410.08 | 412.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:30:00 | 408.00 | 410.17 | 411.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 404.50 | 399.36 | 398.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 404.50 | 399.36 | 398.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 13:15:00 | 407.10 | 403.99 | 402.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 415.20 | 416.71 | 413.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 415.20 | 416.71 | 413.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 411.75 | 415.68 | 413.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 412.05 | 415.68 | 413.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 411.30 | 414.81 | 413.07 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 409.10 | 412.23 | 412.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 405.70 | 410.41 | 411.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 409.40 | 407.91 | 409.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 10:15:00 | 409.40 | 407.91 | 409.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 409.40 | 407.91 | 409.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 410.00 | 407.91 | 409.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 410.30 | 408.39 | 409.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:00:00 | 406.25 | 407.52 | 408.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 405.60 | 407.29 | 408.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 418.25 | 406.13 | 404.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 418.25 | 406.13 | 404.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 425.75 | 415.52 | 410.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 412.00 | 417.84 | 414.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 412.00 | 417.84 | 414.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 412.00 | 417.84 | 414.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 412.00 | 417.84 | 414.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 413.60 | 416.99 | 414.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:00:00 | 415.75 | 416.74 | 414.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 15:15:00 | 415.00 | 416.21 | 415.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 409.90 | 414.17 | 414.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 409.90 | 414.17 | 414.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 408.00 | 412.93 | 413.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 413.10 | 412.97 | 413.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 13:00:00 | 413.10 | 412.97 | 413.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 412.80 | 412.93 | 413.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 414.00 | 412.93 | 413.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 417.55 | 413.86 | 413.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 417.55 | 413.86 | 413.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 418.95 | 414.88 | 414.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 423.75 | 417.33 | 415.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 422.20 | 424.76 | 421.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 422.20 | 424.76 | 421.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 421.85 | 424.17 | 421.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 421.85 | 424.17 | 421.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 420.95 | 423.53 | 421.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 423.30 | 423.53 | 421.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 419.35 | 422.69 | 421.59 | SL hit (close<static) qty=1.00 sl=420.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 430.30 | 434.83 | 435.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 429.35 | 433.74 | 434.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 424.90 | 423.43 | 427.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 424.55 | 423.43 | 427.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 426.50 | 424.05 | 427.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 14:00:00 | 423.00 | 424.84 | 426.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 15:00:00 | 422.75 | 424.42 | 425.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:45:00 | 422.70 | 419.64 | 421.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 13:15:00 | 434.00 | 424.83 | 423.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 13:15:00 | 434.00 | 424.83 | 423.78 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 428.00 | 428.81 | 428.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 416.90 | 426.35 | 427.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 396.25 | 394.35 | 401.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:30:00 | 397.40 | 394.35 | 401.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 400.15 | 397.30 | 400.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 404.90 | 397.30 | 400.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 405.55 | 398.95 | 400.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 404.95 | 398.95 | 400.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 405.55 | 402.09 | 401.94 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 400.20 | 402.13 | 402.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 397.30 | 400.82 | 401.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 400.95 | 400.38 | 401.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:00:00 | 400.95 | 400.38 | 401.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 401.70 | 400.07 | 400.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 401.70 | 400.07 | 400.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 402.80 | 400.62 | 400.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 406.50 | 400.62 | 400.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 405.50 | 401.60 | 401.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 407.70 | 405.94 | 404.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 405.95 | 406.07 | 404.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 13:00:00 | 405.95 | 406.07 | 404.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 404.50 | 406.63 | 405.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 404.50 | 406.63 | 405.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 402.50 | 405.80 | 405.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 402.50 | 405.80 | 405.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 403.20 | 404.69 | 404.71 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 406.55 | 405.06 | 404.88 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 399.00 | 404.03 | 404.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 397.60 | 401.95 | 403.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 397.30 | 397.19 | 399.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 397.65 | 397.19 | 399.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 400.20 | 397.79 | 399.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 400.20 | 397.79 | 399.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 400.40 | 398.31 | 399.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 398.10 | 398.31 | 399.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 403.60 | 399.37 | 399.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 403.60 | 399.37 | 399.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 410.00 | 401.50 | 400.82 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 398.00 | 400.13 | 400.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 392.45 | 398.59 | 399.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 390.55 | 390.46 | 393.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 12:45:00 | 390.70 | 390.46 | 393.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 389.00 | 390.10 | 392.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 387.90 | 389.66 | 392.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 387.10 | 389.15 | 391.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 14:15:00 | 368.50 | 376.13 | 382.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 15:15:00 | 376.60 | 376.23 | 381.59 | SL hit (close>ema200) qty=0.50 sl=376.23 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 382.00 | 381.37 | 381.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 388.60 | 382.82 | 381.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 390.55 | 394.19 | 391.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 390.55 | 394.19 | 391.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 390.55 | 394.19 | 391.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 390.55 | 394.19 | 391.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 390.00 | 393.36 | 391.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 390.00 | 393.36 | 391.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 389.45 | 392.57 | 390.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:15:00 | 391.85 | 392.57 | 390.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 391.95 | 392.45 | 390.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 393.50 | 392.67 | 391.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 394.55 | 392.04 | 391.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 387.70 | 390.40 | 390.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 387.70 | 390.40 | 390.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 09:15:00 | 382.20 | 388.44 | 389.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 385.90 | 385.15 | 387.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 14:00:00 | 385.90 | 385.15 | 387.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 390.35 | 386.19 | 387.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 390.35 | 386.19 | 387.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 393.30 | 387.61 | 388.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:15:00 | 389.90 | 388.24 | 388.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 391.30 | 388.85 | 388.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 10:15:00 | 391.30 | 388.85 | 388.68 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 386.80 | 388.62 | 388.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 383.35 | 387.26 | 388.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 376.85 | 376.80 | 379.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:30:00 | 375.90 | 376.80 | 379.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 365.60 | 365.00 | 367.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:45:00 | 369.50 | 365.00 | 367.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 366.50 | 365.48 | 367.44 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 14:15:00 | 376.80 | 369.50 | 368.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 379.25 | 372.48 | 370.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 10:15:00 | 362.60 | 375.36 | 373.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 10:15:00 | 362.60 | 375.36 | 373.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 362.60 | 375.36 | 373.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 362.60 | 375.36 | 373.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 366.75 | 373.63 | 373.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:30:00 | 362.00 | 373.63 | 373.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 364.80 | 371.87 | 372.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 14:15:00 | 362.90 | 368.78 | 370.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 367.75 | 364.34 | 366.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 367.75 | 364.34 | 366.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 367.75 | 364.34 | 366.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 368.90 | 364.34 | 366.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 367.80 | 365.03 | 366.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 11:30:00 | 366.60 | 365.40 | 366.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:45:00 | 366.30 | 365.96 | 366.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 12:15:00 | 367.35 | 366.74 | 366.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 12:15:00 | 367.95 | 366.98 | 366.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 12:15:00 | 367.95 | 366.98 | 366.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 369.70 | 367.65 | 367.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 377.00 | 377.96 | 375.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 377.00 | 377.96 | 375.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 376.90 | 377.56 | 375.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:00:00 | 379.10 | 377.87 | 375.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 378.75 | 377.96 | 376.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-09 14:15:00 | 417.01 | 407.94 | 401.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 408.85 | 410.39 | 410.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 406.75 | 409.66 | 410.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 409.70 | 409.28 | 409.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 12:15:00 | 409.70 | 409.28 | 409.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 409.70 | 409.28 | 409.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 409.70 | 409.28 | 409.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 408.55 | 409.14 | 409.72 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 412.60 | 410.14 | 410.04 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 408.70 | 410.07 | 410.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 404.50 | 408.70 | 409.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 396.45 | 396.30 | 399.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 396.45 | 396.30 | 399.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 384.80 | 382.66 | 384.73 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 389.85 | 385.25 | 384.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 15:15:00 | 391.65 | 387.43 | 386.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 14:15:00 | 388.25 | 389.73 | 388.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 14:15:00 | 388.25 | 389.73 | 388.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 388.25 | 389.73 | 388.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 388.25 | 389.73 | 388.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 388.60 | 389.51 | 388.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 391.80 | 389.51 | 388.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 387.10 | 387.94 | 387.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 387.10 | 387.94 | 387.98 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 388.80 | 388.04 | 388.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 390.75 | 388.59 | 388.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 390.30 | 390.31 | 389.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 390.30 | 390.31 | 389.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 388.45 | 389.94 | 389.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 389.05 | 389.94 | 389.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 387.50 | 389.45 | 389.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 383.25 | 389.45 | 389.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 383.85 | 388.33 | 388.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 377.45 | 386.15 | 387.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 376.60 | 375.71 | 378.58 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:15:00 | 365.95 | 372.16 | 374.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 362.45 | 357.48 | 361.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 362.45 | 357.48 | 361.77 | SL hit (close>ema400) qty=1.00 sl=361.77 alert=retest1 |

### Cycle 124 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 364.65 | 362.73 | 362.53 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 14:15:00 | 361.70 | 362.72 | 362.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 15:15:00 | 361.20 | 362.41 | 362.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 362.45 | 362.42 | 362.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 362.45 | 362.42 | 362.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 362.45 | 362.42 | 362.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 361.90 | 362.42 | 362.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 360.45 | 362.03 | 362.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 12:30:00 | 359.60 | 361.20 | 361.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:30:00 | 358.85 | 359.92 | 361.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 357.80 | 360.90 | 361.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 341.62 | 345.89 | 348.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 340.91 | 341.90 | 346.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 339.91 | 341.90 | 346.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 13:15:00 | 323.64 | 332.29 | 339.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 126 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 324.15 | 317.72 | 317.26 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 314.25 | 318.56 | 318.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 311.90 | 317.23 | 317.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 13:15:00 | 312.20 | 311.23 | 313.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 13:15:00 | 312.20 | 311.23 | 313.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 312.20 | 311.23 | 313.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 312.20 | 311.23 | 313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 313.85 | 311.75 | 313.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 313.85 | 311.75 | 313.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 313.80 | 312.16 | 313.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 304.15 | 312.16 | 313.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 288.94 | 294.00 | 298.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 293.60 | 293.35 | 296.90 | SL hit (close>ema200) qty=0.50 sl=293.35 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 287.35 | 283.13 | 283.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 288.05 | 285.10 | 284.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 291.05 | 291.38 | 288.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 291.05 | 291.38 | 288.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 290.85 | 291.14 | 289.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 292.65 | 291.01 | 289.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 287.85 | 289.21 | 289.18 | SL hit (close<static) qty=1.00 sl=288.05 alert=retest2 |

### Cycle 129 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 287.35 | 288.84 | 289.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 286.40 | 288.14 | 288.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 261.70 | 255.93 | 261.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 261.70 | 255.93 | 261.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 261.70 | 255.93 | 261.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 261.70 | 255.93 | 261.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 261.35 | 257.01 | 261.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:30:00 | 259.55 | 257.68 | 261.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 263.30 | 260.52 | 261.20 | SL hit (close>static) qty=1.00 sl=263.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 268.40 | 262.57 | 262.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 14:15:00 | 273.25 | 269.70 | 266.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 277.30 | 277.66 | 274.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 277.30 | 277.66 | 274.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 273.60 | 276.69 | 274.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 273.60 | 276.69 | 274.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 270.20 | 275.39 | 273.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 270.20 | 275.39 | 273.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 270.50 | 274.41 | 273.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 267.05 | 274.41 | 273.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 271.45 | 273.07 | 273.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 268.55 | 270.51 | 271.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 270.05 | 266.48 | 267.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 270.05 | 266.48 | 267.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 270.05 | 266.48 | 267.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 270.05 | 266.48 | 267.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 269.80 | 267.14 | 268.10 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 270.80 | 268.59 | 268.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 271.70 | 269.22 | 268.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 291.70 | 295.28 | 292.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 291.70 | 295.28 | 292.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 291.70 | 295.28 | 292.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 291.70 | 295.28 | 292.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 290.25 | 294.28 | 291.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 290.85 | 294.28 | 291.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 289.60 | 293.34 | 291.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 288.00 | 293.34 | 291.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 288.60 | 290.90 | 290.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 289.70 | 290.90 | 290.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 288.75 | 290.47 | 290.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 285.80 | 288.52 | 289.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 287.05 | 286.36 | 288.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 11:00:00 | 287.05 | 286.36 | 288.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 284.45 | 285.08 | 286.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:15:00 | 283.45 | 285.08 | 286.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:00:00 | 283.65 | 284.79 | 286.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:00:00 | 283.15 | 284.46 | 285.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 283.70 | 281.49 | 283.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 283.60 | 281.92 | 283.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 282.50 | 282.26 | 283.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 280.60 | 283.59 | 283.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 283.40 | 283.60 | 283.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 12:15:00 | 285.65 | 284.19 | 284.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 285.65 | 284.19 | 284.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 286.25 | 284.60 | 284.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 283.15 | 289.19 | 287.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 283.15 | 289.19 | 287.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 283.15 | 289.19 | 287.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 283.15 | 289.19 | 287.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 282.40 | 287.83 | 287.33 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 281.15 | 286.50 | 286.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 280.25 | 285.25 | 286.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 269.00 | 267.65 | 274.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 271.80 | 267.65 | 274.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 272.40 | 269.62 | 272.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 272.80 | 269.62 | 272.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 272.85 | 270.26 | 272.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:15:00 | 273.45 | 270.26 | 272.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 273.45 | 270.90 | 272.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 267.70 | 270.90 | 272.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 267.60 | 270.24 | 272.25 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 277.50 | 272.65 | 272.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 279.40 | 274.00 | 272.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 307.10 | 310.54 | 308.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 307.10 | 310.54 | 308.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 307.10 | 310.54 | 308.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 307.10 | 310.54 | 308.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 308.05 | 310.05 | 308.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:00:00 | 308.05 | 310.05 | 308.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 308.35 | 309.71 | 308.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 308.35 | 309.71 | 308.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 309.35 | 309.64 | 308.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:00:00 | 309.95 | 309.70 | 308.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 310.20 | 309.65 | 308.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:15:00 | 310.00 | 309.65 | 308.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:30:00 | 310.55 | 311.92 | 311.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 310.15 | 310.54 | 310.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 310.15 | 310.54 | 310.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 308.85 | 310.12 | 310.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 309.80 | 308.16 | 308.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 309.80 | 308.16 | 308.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 309.80 | 308.16 | 308.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 309.15 | 308.16 | 308.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 309.00 | 308.33 | 308.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 308.00 | 308.43 | 308.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 307.15 | 308.43 | 308.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:00:00 | 308.20 | 308.39 | 308.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 10:15:00 | 318.45 | 310.02 | 309.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 318.45 | 310.02 | 309.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 322.80 | 317.75 | 315.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 317.90 | 319.12 | 316.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 317.90 | 319.12 | 316.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 317.90 | 319.12 | 316.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 318.60 | 319.12 | 316.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 315.35 | 318.37 | 316.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 315.35 | 318.37 | 316.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 313.50 | 317.40 | 316.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 313.50 | 317.40 | 316.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 312.35 | 316.39 | 315.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:30:00 | 312.20 | 316.39 | 315.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 315.05 | 315.42 | 315.46 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 317.80 | 315.89 | 315.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 318.40 | 316.40 | 315.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 329.30 | 329.96 | 324.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 329.30 | 329.96 | 324.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 324.25 | 328.82 | 324.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 324.25 | 328.82 | 324.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 323.00 | 327.66 | 324.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 315.80 | 327.66 | 324.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 318.30 | 325.78 | 323.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 320.85 | 324.41 | 323.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 320.00 | 322.89 | 322.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 12:15:00 | 320.00 | 322.89 | 322.94 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 332.40 | 324.58 | 323.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 341.75 | 336.34 | 333.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 341.25 | 342.05 | 338.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 14:00:00 | 341.25 | 342.05 | 338.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 340.50 | 341.49 | 339.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 345.15 | 341.49 | 339.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-21 12:15:00 | 379.67 | 365.25 | 357.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 375.75 | 378.62 | 378.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 375.10 | 377.50 | 378.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 377.75 | 377.41 | 378.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 377.75 | 377.41 | 378.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 377.75 | 377.41 | 378.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 377.75 | 377.41 | 378.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 380.55 | 378.04 | 378.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 380.55 | 378.04 | 378.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 378.50 | 378.13 | 378.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:15:00 | 377.55 | 378.13 | 378.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 377.85 | 377.49 | 377.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 12:15:00 | 379.10 | 378.12 | 378.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 379.10 | 378.12 | 378.05 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 14:15:00 | 376.40 | 377.72 | 377.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 15:15:00 | 375.05 | 377.18 | 377.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 379.50 | 375.68 | 376.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 379.50 | 375.68 | 376.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 379.50 | 375.68 | 376.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 378.30 | 375.68 | 376.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 380.50 | 376.64 | 376.58 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 368.20 | 376.05 | 376.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 367.30 | 374.30 | 375.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 371.45 | 369.92 | 372.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 371.45 | 369.92 | 372.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 371.45 | 369.92 | 372.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 372.80 | 369.92 | 372.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 372.20 | 370.37 | 372.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 372.25 | 370.37 | 372.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 374.20 | 371.31 | 372.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 374.20 | 371.31 | 372.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 373.15 | 371.68 | 372.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 373.20 | 371.68 | 372.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 373.65 | 372.25 | 372.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 375.10 | 372.25 | 372.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 376.90 | 373.18 | 373.04 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 371.00 | 373.07 | 373.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 369.65 | 372.16 | 372.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 371.85 | 370.43 | 371.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 371.85 | 370.43 | 371.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 371.85 | 370.43 | 371.28 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 379.45 | 372.23 | 372.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 383.50 | 377.85 | 375.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 381.10 | 381.38 | 378.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 15:00:00 | 381.10 | 381.38 | 378.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 378.20 | 381.51 | 380.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 378.80 | 381.51 | 380.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 381.35 | 381.48 | 380.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 383.85 | 381.19 | 380.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 377.35 | 380.39 | 380.12 | SL hit (close<static) qty=1.00 sl=378.20 alert=retest2 |

### Cycle 151 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 372.65 | 378.84 | 379.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 369.70 | 376.03 | 377.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 368.85 | 365.39 | 368.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 14:15:00 | 368.85 | 365.39 | 368.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 368.85 | 365.39 | 368.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 368.85 | 365.39 | 368.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 365.50 | 365.41 | 368.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 367.50 | 365.41 | 368.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 367.80 | 365.89 | 368.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 365.20 | 365.96 | 368.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 364.80 | 365.58 | 367.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 363.60 | 364.75 | 366.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 364.95 | 363.21 | 364.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 360.70 | 362.70 | 364.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:30:00 | 356.15 | 360.53 | 362.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 15:15:00 | 346.94 | 354.12 | 357.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 15:15:00 | 346.56 | 354.12 | 357.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 15:15:00 | 346.70 | 354.12 | 357.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 345.42 | 353.08 | 356.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 347.40 | 346.09 | 350.47 | SL hit (close>ema200) qty=0.50 sl=346.09 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 357.80 | 350.90 | 350.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 365.25 | 358.15 | 356.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 15:15:00 | 366.15 | 366.31 | 363.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:00:00 | 366.60 | 366.37 | 364.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 364.70 | 366.03 | 364.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 364.35 | 366.03 | 364.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 370.70 | 367.27 | 365.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:00:00 | 372.70 | 368.08 | 367.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 372.75 | 368.67 | 367.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:45:00 | 372.35 | 369.18 | 367.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 372.30 | 370.23 | 368.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 371.00 | 372.73 | 371.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 371.00 | 372.73 | 371.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 370.50 | 372.28 | 371.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 370.45 | 372.28 | 371.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 369.95 | 370.86 | 370.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 369.00 | 370.86 | 370.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 365.00 | 369.69 | 370.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 365.00 | 369.69 | 370.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 364.15 | 368.58 | 369.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 12:15:00 | 365.90 | 364.60 | 366.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 13:00:00 | 365.90 | 364.60 | 366.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 364.80 | 364.64 | 366.26 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 370.65 | 367.18 | 367.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 370.95 | 367.93 | 367.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 368.30 | 369.77 | 368.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 368.30 | 369.77 | 368.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 368.30 | 369.77 | 368.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 369.10 | 369.77 | 368.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 367.05 | 369.23 | 368.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:15:00 | 366.40 | 369.23 | 368.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 368.15 | 368.60 | 368.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 368.65 | 368.60 | 368.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 367.05 | 368.29 | 368.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 367.50 | 368.29 | 368.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 367.00 | 368.03 | 368.17 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 371.15 | 368.66 | 368.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 377.55 | 370.44 | 369.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 372.85 | 373.84 | 371.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 372.85 | 373.84 | 371.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 370.90 | 373.25 | 371.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 369.05 | 373.25 | 371.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 371.00 | 372.80 | 371.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 370.50 | 372.80 | 371.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 371.30 | 372.50 | 371.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:15:00 | 370.30 | 372.50 | 371.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 359.40 | 368.95 | 370.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 356.55 | 360.24 | 364.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 356.95 | 354.27 | 356.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 356.95 | 354.27 | 356.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 356.95 | 354.27 | 356.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 356.85 | 354.27 | 356.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 355.50 | 354.52 | 356.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 353.90 | 354.52 | 356.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 354.90 | 354.26 | 355.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 337.15 | 341.65 | 345.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 336.20 | 340.49 | 344.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 340.85 | 340.27 | 343.99 | SL hit (close>ema200) qty=0.50 sl=340.27 alert=retest2 |

### Cycle 158 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 336.90 | 325.73 | 324.70 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 319.70 | 323.87 | 324.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 318.90 | 322.26 | 323.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 323.25 | 314.85 | 315.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 323.25 | 314.85 | 315.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 323.25 | 314.85 | 315.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 323.25 | 314.85 | 315.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 327.85 | 317.45 | 316.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 329.60 | 322.09 | 319.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 329.95 | 330.27 | 327.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 329.95 | 330.27 | 327.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 329.40 | 330.38 | 328.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 329.25 | 330.38 | 328.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 329.80 | 330.13 | 329.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 326.60 | 330.13 | 329.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 327.15 | 329.53 | 328.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 325.90 | 329.53 | 328.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 326.45 | 328.92 | 328.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 326.60 | 328.92 | 328.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 326.55 | 328.12 | 328.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 324.30 | 327.09 | 327.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 329.30 | 327.36 | 327.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 329.30 | 327.36 | 327.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 329.30 | 327.36 | 327.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 330.50 | 327.36 | 327.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 328.95 | 327.68 | 327.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 328.20 | 327.68 | 327.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 329.80 | 328.10 | 328.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 329.80 | 328.10 | 328.05 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 326.80 | 327.95 | 328.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 323.60 | 326.93 | 327.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 12:15:00 | 323.45 | 322.81 | 324.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:00:00 | 323.45 | 322.81 | 324.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 323.00 | 321.70 | 322.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 323.15 | 321.70 | 322.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 322.30 | 321.82 | 322.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 321.45 | 321.86 | 322.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 321.20 | 321.86 | 322.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 323.85 | 322.03 | 322.66 | SL hit (close>static) qty=1.00 sl=323.35 alert=retest2 |

### Cycle 164 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 325.25 | 323.12 | 323.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 328.95 | 324.29 | 323.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 345.70 | 346.01 | 340.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:30:00 | 345.50 | 346.01 | 340.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 344.30 | 347.17 | 344.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 344.30 | 347.17 | 344.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 344.15 | 346.57 | 344.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 344.50 | 346.57 | 344.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 346.95 | 346.64 | 344.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 14:00:00 | 349.50 | 347.02 | 345.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 351.00 | 349.08 | 346.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-23 09:15:00 | 384.45 | 378.36 | 376.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 375.80 | 378.10 | 378.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 374.10 | 376.70 | 377.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 15:15:00 | 373.75 | 373.72 | 375.14 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:15:00 | 367.85 | 373.72 | 375.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 361.20 | 358.18 | 360.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 361.20 | 358.18 | 360.34 | SL hit (close>ema400) qty=1.00 sl=360.34 alert=retest1 |

### Cycle 166 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 365.00 | 361.39 | 361.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 367.00 | 363.24 | 362.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 369.95 | 370.77 | 367.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 369.95 | 370.77 | 367.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 380.90 | 376.52 | 373.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 383.80 | 376.52 | 373.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 383.00 | 377.21 | 374.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:45:00 | 381.45 | 379.04 | 375.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 381.60 | 380.23 | 378.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 379.00 | 381.23 | 380.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 378.05 | 381.23 | 380.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 377.60 | 380.50 | 379.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 377.60 | 380.50 | 379.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 379.35 | 380.27 | 379.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:45:00 | 380.45 | 380.04 | 379.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 372.85 | 378.44 | 379.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 372.85 | 378.44 | 379.14 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 381.60 | 377.17 | 377.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 386.55 | 382.14 | 379.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 418.40 | 418.63 | 412.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 412.75 | 416.62 | 412.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 412.75 | 416.62 | 412.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 411.60 | 416.62 | 412.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 410.65 | 415.43 | 412.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 409.90 | 415.43 | 412.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 411.60 | 414.66 | 412.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 410.10 | 414.66 | 412.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 416.05 | 414.51 | 412.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 427.85 | 413.20 | 412.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 452.45 | 454.20 | 454.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 452.45 | 454.20 | 454.25 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 459.75 | 454.96 | 454.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 463.20 | 457.63 | 455.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 466.05 | 468.20 | 464.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 466.05 | 468.20 | 464.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 466.05 | 468.20 | 464.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 466.05 | 468.20 | 464.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 465.45 | 467.65 | 464.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 470.50 | 467.65 | 464.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 464.00 | 466.92 | 464.23 | SL hit (close<static) qty=1.00 sl=464.05 alert=retest2 |

### Cycle 171 — SELL (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 15:15:00 | 459.95 | 462.84 | 463.13 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 467.20 | 463.71 | 463.50 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 461.45 | 463.47 | 463.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 458.00 | 461.40 | 462.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 450.40 | 448.92 | 452.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:30:00 | 449.15 | 448.92 | 452.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 451.85 | 450.00 | 452.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 451.50 | 450.20 | 452.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 455.50 | 451.23 | 452.59 | SL hit (close>static) qty=1.00 sl=453.45 alert=retest2 |

### Cycle 174 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 454.40 | 448.11 | 447.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 457.05 | 451.67 | 449.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 447.95 | 451.54 | 449.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 447.95 | 451.54 | 449.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 447.95 | 451.54 | 449.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 447.95 | 451.54 | 449.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 447.50 | 450.73 | 449.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 447.10 | 450.73 | 449.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 446.85 | 449.96 | 449.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 446.85 | 449.96 | 449.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 447.80 | 448.99 | 449.06 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 459.75 | 450.28 | 449.48 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 449.45 | 450.13 | 450.21 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 452.05 | 450.51 | 450.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 455.00 | 451.41 | 450.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 466.20 | 470.24 | 465.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 466.20 | 470.24 | 465.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 466.20 | 470.24 | 465.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 461.50 | 470.24 | 465.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 468.20 | 469.83 | 465.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 465.50 | 469.83 | 465.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 464.65 | 468.80 | 465.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 464.80 | 468.80 | 465.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 464.35 | 467.91 | 465.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 463.85 | 467.91 | 465.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 466.90 | 467.71 | 465.75 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 455.30 | 463.86 | 464.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 452.25 | 461.53 | 463.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 452.45 | 448.81 | 452.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 452.45 | 448.81 | 452.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 452.45 | 448.81 | 452.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 451.80 | 448.81 | 452.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 450.55 | 449.16 | 452.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 449.00 | 449.68 | 452.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 454.00 | 448.98 | 450.53 | SL hit (close>static) qty=1.00 sl=453.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 453.15 | 451.60 | 451.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 460.40 | 453.36 | 452.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 465.90 | 466.63 | 463.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 465.90 | 466.63 | 463.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 465.20 | 466.35 | 463.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 462.85 | 466.35 | 463.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 463.35 | 465.75 | 463.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 463.35 | 465.75 | 463.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 464.85 | 465.57 | 463.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:15:00 | 465.65 | 465.57 | 463.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 465.55 | 465.56 | 463.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 455.75 | 463.41 | 463.35 | SL hit (close<static) qty=1.00 sl=462.90 alert=retest2 |

### Cycle 181 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 455.95 | 461.92 | 462.67 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 480.00 | 463.79 | 462.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 488.25 | 468.68 | 465.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 507.85 | 509.08 | 502.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 15:00:00 | 507.85 | 509.08 | 502.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 502.85 | 507.56 | 502.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 502.50 | 507.56 | 502.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 501.50 | 506.35 | 502.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 501.50 | 506.35 | 502.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 501.75 | 505.43 | 502.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 500.80 | 505.43 | 502.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 499.95 | 503.57 | 502.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:45:00 | 499.60 | 503.57 | 502.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 505.00 | 503.42 | 502.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 504.25 | 503.42 | 502.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 499.50 | 502.64 | 502.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 499.50 | 502.64 | 502.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 501.65 | 502.44 | 502.15 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 499.95 | 501.94 | 501.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 498.60 | 501.27 | 501.64 | Break + close below crossover candle low |

### Cycle 184 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 504.80 | 501.98 | 501.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 506.90 | 504.00 | 503.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 502.10 | 504.46 | 503.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 14:15:00 | 502.10 | 504.46 | 503.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 502.10 | 504.46 | 503.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 502.10 | 504.46 | 503.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 503.35 | 504.24 | 503.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 511.20 | 504.24 | 503.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 508.45 | 511.10 | 511.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 508.45 | 511.10 | 511.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 503.05 | 508.89 | 510.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 510.15 | 506.93 | 508.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 510.15 | 506.93 | 508.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 510.15 | 506.93 | 508.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 510.20 | 506.93 | 508.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 510.15 | 507.57 | 508.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 511.50 | 507.57 | 508.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 516.55 | 510.59 | 509.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 10:15:00 | 522.10 | 515.16 | 512.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 513.65 | 515.49 | 513.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 13:15:00 | 513.65 | 515.49 | 513.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 513.65 | 515.49 | 513.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 512.80 | 515.49 | 513.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 507.10 | 513.81 | 512.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 507.10 | 513.81 | 512.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 505.35 | 512.12 | 512.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 15:15:00 | 504.85 | 507.78 | 509.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 501.70 | 501.41 | 505.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:30:00 | 501.70 | 501.41 | 505.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 505.00 | 502.11 | 504.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 506.90 | 502.11 | 504.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 503.70 | 502.43 | 504.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 501.85 | 502.43 | 504.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 502.35 | 502.71 | 504.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 501.60 | 502.73 | 504.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 509.05 | 504.56 | 504.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 509.05 | 504.56 | 504.46 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 502.50 | 504.41 | 504.57 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 506.35 | 504.61 | 504.47 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 500.50 | 503.85 | 504.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 496.40 | 501.44 | 503.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 503.75 | 500.29 | 501.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 11:15:00 | 503.75 | 500.29 | 501.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 503.75 | 500.29 | 501.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 503.75 | 500.29 | 501.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 507.70 | 501.77 | 502.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 507.70 | 501.77 | 502.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 13:15:00 | 508.80 | 503.18 | 502.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 14:15:00 | 510.80 | 504.70 | 503.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 518.15 | 518.45 | 513.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 518.15 | 518.45 | 513.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 515.00 | 517.92 | 514.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 514.95 | 517.92 | 514.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 510.55 | 516.45 | 514.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 510.55 | 516.45 | 514.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 503.00 | 513.76 | 513.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 504.35 | 513.76 | 513.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 508.30 | 512.67 | 512.71 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 512.15 | 506.33 | 506.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 519.55 | 510.68 | 508.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 511.75 | 514.44 | 511.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 511.75 | 514.44 | 511.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 511.75 | 514.44 | 511.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 511.75 | 514.44 | 511.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 500.00 | 511.56 | 510.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 508.25 | 511.56 | 510.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 500.00 | 509.24 | 509.50 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 12:15:00 | 512.35 | 509.28 | 509.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 13:15:00 | 514.10 | 510.25 | 509.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 540.70 | 543.72 | 536.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 540.70 | 543.72 | 536.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 537.10 | 540.45 | 537.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:30:00 | 536.85 | 540.45 | 537.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 536.10 | 539.58 | 537.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 536.10 | 539.58 | 537.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 535.20 | 538.70 | 536.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 526.75 | 538.70 | 536.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 526.25 | 534.35 | 535.07 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 543.70 | 536.57 | 535.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 569.00 | 543.60 | 539.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 14:15:00 | 554.50 | 559.01 | 550.06 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:15:00 | 583.40 | 558.61 | 550.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:30:00 | 569.85 | 564.72 | 555.79 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:15:00 | 598.34 | 573.60 | 563.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 587.15 | 593.25 | 582.94 | SL hit (close<ema200) qty=0.50 sl=593.25 alert=retest1 |

### Cycle 199 — SELL (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 15:15:00 | 582.55 | 585.74 | 585.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 09:15:00 | 576.20 | 583.83 | 585.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 574.90 | 574.19 | 578.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:30:00 | 574.05 | 574.19 | 578.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 571.85 | 570.83 | 574.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 576.50 | 570.83 | 574.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 577.20 | 572.10 | 574.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 577.20 | 572.10 | 574.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 574.05 | 572.49 | 574.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:30:00 | 581.50 | 572.49 | 574.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 567.90 | 571.57 | 573.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 570.75 | 571.57 | 573.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 533.40 | 534.35 | 539.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 531.35 | 536.56 | 538.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 504.78 | 513.59 | 523.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 478.22 | 505.29 | 517.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 200 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 449.00 | 440.26 | 440.08 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 436.55 | 439.58 | 439.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 435.00 | 438.66 | 439.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 15:15:00 | 431.10 | 430.40 | 433.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:15:00 | 424.25 | 430.40 | 433.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 403.04 | 411.83 | 420.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 13:15:00 | 413.25 | 410.67 | 417.58 | SL hit (close>ema200) qty=0.50 sl=410.67 alert=retest1 |

### Cycle 202 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 422.70 | 418.57 | 418.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 429.95 | 420.85 | 419.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 424.05 | 429.88 | 425.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 424.05 | 429.88 | 425.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 424.05 | 429.88 | 425.62 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 413.80 | 422.40 | 423.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 412.10 | 420.34 | 422.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 419.55 | 419.41 | 421.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 419.55 | 419.41 | 421.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 419.55 | 419.41 | 421.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 420.60 | 419.41 | 421.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 416.35 | 418.84 | 420.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 413.65 | 418.84 | 420.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 405.70 | 418.26 | 419.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 392.97 | 404.92 | 412.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 404.20 | 400.47 | 407.40 | SL hit (close>ema200) qty=0.50 sl=400.47 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 407.45 | 405.71 | 405.64 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 396.15 | 404.09 | 404.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 386.00 | 397.27 | 400.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 399.45 | 388.62 | 393.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 399.45 | 388.62 | 393.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 399.45 | 388.62 | 393.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 399.45 | 388.62 | 393.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 393.80 | 389.66 | 393.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 392.25 | 392.70 | 393.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 383.40 | 393.15 | 393.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 394.20 | 389.80 | 389.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 394.20 | 389.80 | 389.79 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 388.15 | 389.64 | 389.76 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 391.80 | 390.08 | 389.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 428.15 | 398.76 | 394.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 419.50 | 419.92 | 413.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 429.30 | 419.92 | 413.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 419.80 | 427.38 | 421.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 419.80 | 427.38 | 421.96 | SL hit (close<ema400) qty=1.00 sl=421.96 alert=retest1 |

### Cycle 209 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 419.35 | 423.38 | 423.75 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 435.50 | 425.30 | 424.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 437.65 | 427.77 | 425.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 14:15:00 | 428.60 | 429.24 | 426.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 14:30:00 | 429.05 | 429.24 | 426.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 428.50 | 429.09 | 427.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 426.45 | 429.09 | 427.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 423.55 | 427.98 | 426.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 422.85 | 427.98 | 426.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 423.50 | 427.09 | 426.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:45:00 | 422.85 | 427.09 | 426.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 423.10 | 425.76 | 425.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 421.20 | 424.85 | 425.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 425.95 | 423.38 | 424.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 425.95 | 423.38 | 424.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 425.95 | 423.38 | 424.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 425.95 | 423.38 | 424.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 426.00 | 423.90 | 424.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 426.40 | 423.90 | 424.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 425.10 | 424.39 | 424.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:15:00 | 425.75 | 424.39 | 424.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 426.70 | 424.85 | 424.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 426.70 | 424.85 | 424.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 423.75 | 424.63 | 424.82 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 427.50 | 425.01 | 424.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 429.00 | 425.81 | 425.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 14:15:00 | 426.55 | 426.85 | 425.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 14:15:00 | 426.55 | 426.85 | 425.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 426.55 | 426.85 | 425.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:00:00 | 426.55 | 426.85 | 425.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 422.55 | 426.03 | 425.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 422.15 | 426.03 | 425.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 422.30 | 425.28 | 425.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 421.30 | 423.65 | 424.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 403.50 | 403.26 | 409.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 403.50 | 403.26 | 409.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 404.75 | 404.38 | 407.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 402.90 | 404.50 | 407.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 415.45 | 405.39 | 406.48 | SL hit (close>static) qty=1.00 sl=408.75 alert=retest2 |

### Cycle 214 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 410.00 | 407.41 | 407.25 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 397.60 | 405.66 | 406.56 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 415.35 | 406.24 | 405.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 417.50 | 408.49 | 406.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 410.00 | 411.39 | 408.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 410.00 | 411.39 | 408.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 410.00 | 411.39 | 408.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 410.00 | 411.39 | 408.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 407.45 | 410.62 | 409.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 407.75 | 410.62 | 409.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 408.50 | 410.19 | 409.01 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 405.75 | 408.03 | 408.26 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 13:45:00 | 407.65 | 2024-04-22 09:15:00 | 418.35 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-04-24 11:30:00 | 407.50 | 2024-04-25 09:15:00 | 412.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-04-24 12:30:00 | 407.40 | 2024-04-25 09:15:00 | 412.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-04-24 13:45:00 | 407.25 | 2024-04-25 09:15:00 | 412.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-05-08 14:15:00 | 399.55 | 2024-05-10 09:15:00 | 379.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 398.00 | 2024-05-10 09:15:00 | 379.71 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2024-05-09 10:15:00 | 399.70 | 2024-05-10 09:15:00 | 380.14 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2024-05-09 12:00:00 | 400.15 | 2024-05-13 09:15:00 | 378.10 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2024-05-08 14:15:00 | 399.55 | 2024-05-13 13:15:00 | 381.90 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2024-05-09 09:15:00 | 398.00 | 2024-05-13 13:15:00 | 381.90 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-05-09 10:15:00 | 399.70 | 2024-05-13 13:15:00 | 381.90 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2024-05-09 12:00:00 | 400.15 | 2024-05-13 13:15:00 | 381.90 | STOP_HIT | 0.50 | 4.56% |
| BUY | retest2 | 2024-05-27 09:15:00 | 424.00 | 2024-05-28 09:15:00 | 414.25 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-06-03 11:00:00 | 408.80 | 2024-06-03 12:15:00 | 411.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-06-11 12:15:00 | 408.85 | 2024-06-12 11:15:00 | 404.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-11 13:15:00 | 408.50 | 2024-06-12 11:15:00 | 404.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-06-12 09:15:00 | 410.40 | 2024-06-12 11:15:00 | 404.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-06-19 11:15:00 | 393.40 | 2024-06-25 14:15:00 | 399.60 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-06-19 12:30:00 | 393.65 | 2024-06-25 14:15:00 | 399.60 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-06-19 15:00:00 | 393.40 | 2024-06-25 14:15:00 | 399.60 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-06-20 09:45:00 | 393.45 | 2024-06-25 14:15:00 | 399.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-06-24 09:30:00 | 391.25 | 2024-06-25 15:15:00 | 403.40 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-06-25 11:30:00 | 392.05 | 2024-06-25 15:15:00 | 403.40 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-06-25 13:00:00 | 391.20 | 2024-06-25 15:15:00 | 403.40 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2024-06-25 13:45:00 | 391.60 | 2024-06-25 15:15:00 | 403.40 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-07-08 10:30:00 | 461.65 | 2024-07-09 09:15:00 | 445.30 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2024-07-08 11:15:00 | 459.60 | 2024-07-09 09:15:00 | 445.30 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-07-25 09:45:00 | 435.65 | 2024-07-26 13:15:00 | 446.10 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-07-25 11:15:00 | 437.85 | 2024-07-26 13:15:00 | 446.10 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-07-25 13:15:00 | 437.85 | 2024-07-26 13:15:00 | 446.10 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-07-26 09:15:00 | 437.05 | 2024-07-26 13:15:00 | 446.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-08 14:30:00 | 408.70 | 2024-08-19 10:15:00 | 404.50 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2024-08-09 09:30:00 | 407.85 | 2024-08-19 10:15:00 | 404.50 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2024-08-12 09:30:00 | 408.00 | 2024-08-19 10:15:00 | 404.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-08-28 15:00:00 | 406.25 | 2024-09-04 09:15:00 | 418.25 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-08-29 09:15:00 | 405.60 | 2024-09-04 09:15:00 | 418.25 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-09-06 12:00:00 | 415.75 | 2024-09-09 10:15:00 | 409.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-09-06 15:15:00 | 415.00 | 2024-09-09 10:15:00 | 409.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-12 09:15:00 | 423.30 | 2024-09-12 09:15:00 | 419.35 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-09-12 12:15:00 | 422.60 | 2024-09-18 10:15:00 | 430.30 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2024-09-12 13:15:00 | 422.35 | 2024-09-18 10:15:00 | 430.30 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2024-09-12 14:45:00 | 422.75 | 2024-09-18 10:15:00 | 430.30 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2024-09-13 13:30:00 | 440.45 | 2024-09-18 10:15:00 | 430.30 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-09-17 11:15:00 | 437.20 | 2024-09-18 10:15:00 | 430.30 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-09-17 11:45:00 | 436.95 | 2024-09-18 10:15:00 | 430.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-09-23 14:00:00 | 423.00 | 2024-09-25 13:15:00 | 434.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-09-23 15:00:00 | 422.75 | 2024-09-25 13:15:00 | 434.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-09-25 09:45:00 | 422.70 | 2024-09-25 13:15:00 | 434.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-10-24 11:00:00 | 387.90 | 2024-10-25 14:15:00 | 368.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 11:00:00 | 387.90 | 2024-10-25 15:15:00 | 376.60 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2024-10-24 12:00:00 | 387.10 | 2024-10-29 15:15:00 | 382.00 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2024-11-04 14:00:00 | 393.50 | 2024-11-05 12:15:00 | 387.70 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-11-05 09:45:00 | 394.55 | 2024-11-05 12:15:00 | 387.70 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-07 10:15:00 | 389.90 | 2024-11-07 10:15:00 | 391.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-11-25 11:30:00 | 366.60 | 2024-11-26 12:15:00 | 367.95 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-11-26 09:45:00 | 366.30 | 2024-11-26 12:15:00 | 367.95 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-11-26 12:15:00 | 367.35 | 2024-11-26 12:15:00 | 367.95 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-11-29 12:00:00 | 379.10 | 2024-12-09 14:15:00 | 417.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 13:15:00 | 378.75 | 2024-12-09 14:15:00 | 416.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-01 09:15:00 | 391.80 | 2025-01-02 09:15:00 | 387.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2025-01-10 09:15:00 | 365.95 | 2025-01-14 09:15:00 | 362.45 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-01-17 12:30:00 | 359.60 | 2025-01-24 14:15:00 | 341.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 09:30:00 | 358.85 | 2025-01-27 09:15:00 | 340.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 357.80 | 2025-01-27 09:15:00 | 339.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 12:30:00 | 359.60 | 2025-01-27 13:15:00 | 323.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 09:30:00 | 358.85 | 2025-01-27 14:15:00 | 322.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 357.80 | 2025-01-27 14:15:00 | 322.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 304.15 | 2025-02-12 09:15:00 | 288.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 304.15 | 2025-02-12 12:15:00 | 293.60 | STOP_HIT | 0.50 | 3.47% |
| BUY | retest2 | 2025-02-21 15:00:00 | 292.65 | 2025-02-24 14:15:00 | 287.85 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-03-04 12:30:00 | 259.55 | 2025-03-05 11:15:00 | 263.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-03-28 10:15:00 | 283.45 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-03-28 11:00:00 | 283.65 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-03-28 12:00:00 | 283.15 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-04-01 10:15:00 | 283.70 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-01 11:30:00 | 282.50 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-02 09:15:00 | 280.60 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-04-02 11:15:00 | 283.40 | 2025-04-02 12:15:00 | 285.65 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-04-23 14:00:00 | 309.95 | 2025-04-25 14:15:00 | 310.15 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-04-23 14:45:00 | 310.20 | 2025-04-25 14:15:00 | 310.15 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-04-23 15:15:00 | 310.00 | 2025-04-25 14:15:00 | 310.15 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-04-25 10:30:00 | 310.55 | 2025-04-25 14:15:00 | 310.15 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-04-29 11:45:00 | 308.00 | 2025-04-30 10:15:00 | 318.45 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-04-29 12:15:00 | 307.15 | 2025-04-30 10:15:00 | 318.45 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-04-29 13:00:00 | 308.20 | 2025-04-30 10:15:00 | 318.45 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-05-09 11:15:00 | 320.85 | 2025-05-09 12:15:00 | 320.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-16 09:15:00 | 345.15 | 2025-05-21 12:15:00 | 379.67 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-28 13:15:00 | 377.55 | 2025-05-29 12:15:00 | 379.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-05-29 10:30:00 | 377.85 | 2025-05-29 12:15:00 | 379.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-06-12 10:15:00 | 383.85 | 2025-06-12 12:15:00 | 377.35 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-06-17 12:00:00 | 365.20 | 2025-06-20 15:15:00 | 346.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:30:00 | 364.80 | 2025-06-20 15:15:00 | 346.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 10:15:00 | 363.60 | 2025-06-20 15:15:00 | 346.70 | PARTIAL | 0.50 | 4.65% |
| SELL | retest2 | 2025-06-19 09:45:00 | 364.95 | 2025-06-23 09:15:00 | 345.42 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-06-17 12:00:00 | 365.20 | 2025-06-24 09:15:00 | 347.40 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-06-17 12:30:00 | 364.80 | 2025-06-24 09:15:00 | 347.40 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-06-18 10:15:00 | 363.60 | 2025-06-24 09:15:00 | 347.40 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-06-19 09:45:00 | 364.95 | 2025-06-24 09:15:00 | 347.40 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2025-06-19 12:30:00 | 356.15 | 2025-06-25 10:15:00 | 357.80 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-07-08 13:00:00 | 372.70 | 2025-07-11 09:15:00 | 365.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-08 14:15:00 | 372.75 | 2025-07-11 09:15:00 | 365.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-07-08 14:45:00 | 372.35 | 2025-07-11 09:15:00 | 365.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-09 11:15:00 | 372.30 | 2025-07-11 09:15:00 | 365.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-07-24 11:15:00 | 353.90 | 2025-07-29 09:15:00 | 337.15 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-07-24 14:45:00 | 354.90 | 2025-07-29 10:15:00 | 336.20 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-07-24 11:15:00 | 353.90 | 2025-07-29 12:15:00 | 340.85 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-07-24 14:45:00 | 354.90 | 2025-07-29 12:15:00 | 340.85 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-08-25 11:15:00 | 328.20 | 2025-08-25 11:15:00 | 329.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-08-29 14:30:00 | 321.45 | 2025-09-01 09:15:00 | 323.85 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-08-29 15:00:00 | 321.20 | 2025-09-01 09:15:00 | 323.85 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-05 14:00:00 | 349.50 | 2025-09-23 09:15:00 | 384.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-08 09:30:00 | 351.00 | 2025-09-23 09:15:00 | 386.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-09-26 09:15:00 | 367.85 | 2025-10-01 09:15:00 | 361.20 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2025-10-08 10:15:00 | 383.80 | 2025-10-14 09:15:00 | 372.85 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-10-08 11:15:00 | 383.00 | 2025-10-14 09:15:00 | 372.85 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-10-08 13:45:00 | 381.45 | 2025-10-14 09:15:00 | 372.85 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-10 09:15:00 | 381.60 | 2025-10-14 09:15:00 | 372.85 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-10-13 14:45:00 | 380.45 | 2025-10-14 09:15:00 | 372.85 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-28 09:15:00 | 427.85 | 2025-11-11 13:15:00 | 452.45 | STOP_HIT | 1.00 | 5.75% |
| BUY | retest2 | 2025-11-14 09:15:00 | 470.50 | 2025-11-14 09:15:00 | 464.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-20 14:30:00 | 451.50 | 2025-11-21 09:15:00 | 455.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-21 13:00:00 | 451.45 | 2025-11-26 11:15:00 | 454.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-10 14:00:00 | 449.00 | 2025-12-11 11:15:00 | 454.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-17 13:15:00 | 465.65 | 2025-12-18 09:15:00 | 455.75 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-12-17 14:00:00 | 465.55 | 2025-12-18 09:15:00 | 455.75 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-01-01 09:15:00 | 511.20 | 2026-01-05 13:15:00 | 508.45 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-13 10:30:00 | 501.85 | 2026-01-14 12:15:00 | 509.05 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-13 12:45:00 | 502.35 | 2026-01-14 12:15:00 | 509.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-13 14:15:00 | 501.60 | 2026-01-14 12:15:00 | 509.05 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2026-02-10 09:15:00 | 583.40 | 2026-02-11 09:15:00 | 598.34 | PARTIAL | 0.50 | 2.56% |
| BUY | retest1 | 2026-02-10 09:15:00 | 583.40 | 2026-02-12 11:15:00 | 587.15 | STOP_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-10 11:30:00 | 569.85 | 2026-02-12 14:15:00 | 582.90 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2026-02-13 11:30:00 | 591.15 | 2026-02-16 15:15:00 | 582.55 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-02-16 09:15:00 | 599.20 | 2026-02-16 15:15:00 | 582.55 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-16 13:15:00 | 588.40 | 2026-02-16 15:15:00 | 582.55 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-02-26 10:15:00 | 531.35 | 2026-02-27 14:15:00 | 504.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:15:00 | 531.35 | 2026-03-02 09:15:00 | 478.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-03-13 09:15:00 | 424.25 | 2026-03-16 10:15:00 | 403.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-13 09:15:00 | 424.25 | 2026-03-16 13:15:00 | 413.25 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2026-03-17 12:15:00 | 411.95 | 2026-03-17 14:15:00 | 420.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-03-20 12:15:00 | 413.65 | 2026-03-23 12:15:00 | 392.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 413.65 | 2026-03-24 09:15:00 | 404.20 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-03-23 09:15:00 | 405.70 | 2026-03-25 13:15:00 | 407.45 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-03-24 09:30:00 | 409.80 | 2026-03-25 13:15:00 | 407.45 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2026-04-01 13:30:00 | 392.25 | 2026-04-06 14:15:00 | 394.20 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-04-02 09:15:00 | 383.40 | 2026-04-06 14:15:00 | 394.20 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2026-04-10 09:15:00 | 429.30 | 2026-04-13 09:15:00 | 419.80 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-04-13 10:45:00 | 424.00 | 2026-04-16 12:15:00 | 419.35 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-04-15 09:15:00 | 432.85 | 2026-04-16 12:15:00 | 419.35 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-04-16 11:00:00 | 424.35 | 2026-04-16 12:15:00 | 419.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-04-28 11:15:00 | 402.90 | 2026-04-29 09:15:00 | 415.45 | STOP_HIT | 1.00 | -3.11% |
