# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 727.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 143 |
| ALERT2 | 139 |
| ALERT2_SKIP | 88 |
| ALERT3 | 317 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 150 |
| PARTIAL | 25 |
| TARGET_HIT | 22 |
| STOP_HIT | 131 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 178 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 71 / 107
- **Target hits / Stop hits / Partials:** 22 / 131 / 25
- **Avg / median % per leg:** 1.18% / -0.80%
- **Sum % (uncompounded):** 210.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 17 | 26.6% | 13 | 51 | 0 | 0.73% | 46.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 64 | 17 | 26.6% | 13 | 51 | 0 | 0.73% | 46.7% |
| SELL (all) | 114 | 54 | 47.4% | 9 | 80 | 25 | 1.44% | 163.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.25% | -3.7% |
| SELL @ 3rd Alert (retest2) | 111 | 54 | 48.6% | 9 | 77 | 25 | 1.51% | 167.7% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.25% | -3.7% |
| retest2 (combined) | 175 | 71 | 40.6% | 22 | 128 | 25 | 1.23% | 214.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 124.30 | 125.76 | 125.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 12:15:00 | 120.68 | 124.75 | 125.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 10:15:00 | 124.10 | 120.30 | 121.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 10:15:00 | 124.10 | 120.30 | 121.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 124.10 | 120.30 | 121.47 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 122.93 | 122.08 | 122.04 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 12:15:00 | 121.28 | 121.98 | 122.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 14:15:00 | 120.95 | 121.72 | 121.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 12:15:00 | 121.28 | 120.95 | 121.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 12:15:00 | 121.28 | 120.95 | 121.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 121.28 | 120.95 | 121.36 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 09:15:00 | 123.78 | 120.97 | 120.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 133.82 | 124.99 | 123.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 15:15:00 | 129.70 | 129.85 | 126.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 13:15:00 | 146.68 | 147.96 | 146.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 146.68 | 147.96 | 146.41 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 160.05 | 160.94 | 160.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 13:15:00 | 159.75 | 160.44 | 160.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 14:15:00 | 160.95 | 160.54 | 160.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 160.95 | 160.54 | 160.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 160.95 | 160.54 | 160.72 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 10:15:00 | 159.15 | 155.65 | 155.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 11:15:00 | 160.38 | 156.60 | 155.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 09:15:00 | 156.73 | 157.72 | 156.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 156.73 | 157.72 | 156.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 156.73 | 157.72 | 156.85 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 155.55 | 156.39 | 156.42 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 10:15:00 | 157.15 | 156.44 | 156.35 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 12:15:00 | 155.78 | 156.27 | 156.29 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 15:15:00 | 157.28 | 156.39 | 156.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 162.75 | 157.66 | 156.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 159.60 | 159.90 | 158.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 10:15:00 | 159.82 | 160.17 | 159.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 159.82 | 160.17 | 159.03 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 166.85 | 167.75 | 167.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 15:15:00 | 166.08 | 167.42 | 167.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 169.08 | 166.12 | 166.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 169.08 | 166.12 | 166.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 169.08 | 166.12 | 166.59 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 11:15:00 | 169.45 | 167.28 | 167.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 12:15:00 | 172.15 | 168.25 | 167.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 09:15:00 | 168.33 | 169.11 | 168.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 168.33 | 169.11 | 168.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 168.33 | 169.11 | 168.25 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 14:15:00 | 181.00 | 188.96 | 189.58 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 10:15:00 | 190.50 | 188.83 | 188.62 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 14:15:00 | 186.95 | 188.54 | 188.57 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 190.43 | 188.75 | 188.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 11:15:00 | 195.90 | 190.63 | 189.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 15:15:00 | 198.98 | 200.29 | 196.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 197.20 | 202.42 | 201.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 197.20 | 202.42 | 201.86 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 10:15:00 | 193.08 | 200.55 | 201.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 11:15:00 | 190.85 | 198.61 | 200.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 12:15:00 | 194.88 | 194.80 | 196.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 14:15:00 | 196.60 | 195.39 | 196.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 196.60 | 195.39 | 196.69 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 197.43 | 197.21 | 197.19 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 15:15:00 | 196.90 | 197.18 | 197.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 12:15:00 | 195.88 | 196.85 | 197.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 194.20 | 193.90 | 194.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 194.20 | 193.90 | 194.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 194.20 | 193.90 | 194.88 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 13:15:00 | 194.53 | 194.18 | 194.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 14:15:00 | 195.50 | 194.44 | 194.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 12:15:00 | 198.73 | 200.13 | 198.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 12:15:00 | 198.73 | 200.13 | 198.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 198.73 | 200.13 | 198.34 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 15:15:00 | 207.38 | 208.34 | 208.37 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 10:15:00 | 209.00 | 208.51 | 208.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 13:15:00 | 210.78 | 208.95 | 208.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 09:15:00 | 209.50 | 209.57 | 209.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 209.50 | 209.57 | 209.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 209.50 | 209.57 | 209.06 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 12:15:00 | 206.35 | 208.41 | 208.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 14:15:00 | 205.00 | 207.46 | 208.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 11:15:00 | 207.78 | 206.47 | 207.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 11:15:00 | 207.78 | 206.47 | 207.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 207.78 | 206.47 | 207.33 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 15:15:00 | 209.80 | 208.15 | 207.93 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 204.23 | 207.36 | 207.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 197.25 | 202.49 | 204.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 13:15:00 | 199.93 | 199.55 | 202.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 203.15 | 200.49 | 201.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 203.15 | 200.49 | 201.85 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 10:15:00 | 204.15 | 202.56 | 202.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 207.73 | 204.61 | 203.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 13:15:00 | 205.40 | 206.34 | 205.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 13:15:00 | 205.40 | 206.34 | 205.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 205.40 | 206.34 | 205.06 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 15:15:00 | 219.75 | 220.60 | 220.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 09:15:00 | 218.98 | 220.27 | 220.53 | Break + close below crossover candle low |

### Cycle 28 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 224.00 | 221.02 | 220.84 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 220.73 | 222.17 | 222.20 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 223.00 | 222.34 | 222.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 10:15:00 | 226.28 | 223.13 | 222.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 223.60 | 224.67 | 223.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 223.60 | 224.67 | 223.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 223.60 | 224.67 | 223.81 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 13:15:00 | 221.93 | 223.42 | 223.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 14:15:00 | 220.08 | 222.75 | 223.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 215.78 | 215.33 | 217.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 11:15:00 | 217.35 | 215.70 | 217.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 217.35 | 215.70 | 217.27 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 12:15:00 | 216.33 | 214.66 | 214.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 221.98 | 216.63 | 215.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 11:15:00 | 218.50 | 219.50 | 218.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 11:15:00 | 218.50 | 219.50 | 218.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 218.50 | 219.50 | 218.17 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 221.20 | 223.15 | 223.38 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 226.50 | 223.36 | 222.98 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 217.00 | 223.50 | 223.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 210.95 | 219.05 | 221.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 204.25 | 203.27 | 208.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 211.30 | 205.15 | 208.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 211.30 | 205.15 | 208.09 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 12:15:00 | 207.60 | 204.99 | 204.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 09:15:00 | 210.75 | 206.86 | 206.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 14:15:00 | 207.50 | 208.05 | 207.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 10:15:00 | 208.48 | 208.26 | 207.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 208.48 | 208.26 | 207.42 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 200.80 | 206.86 | 206.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 12:15:00 | 198.65 | 200.53 | 202.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 199.80 | 199.09 | 201.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 199.80 | 199.09 | 201.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 199.80 | 199.09 | 201.08 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 13:15:00 | 192.95 | 191.66 | 191.56 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 13:15:00 | 190.28 | 191.40 | 191.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 11:15:00 | 189.40 | 190.59 | 190.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 13:15:00 | 189.48 | 189.14 | 189.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 13:15:00 | 189.48 | 189.14 | 189.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 189.48 | 189.14 | 189.82 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 192.40 | 190.43 | 190.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 197.33 | 192.76 | 191.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 196.10 | 197.00 | 195.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 13:15:00 | 196.15 | 196.75 | 195.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 13:15:00 | 196.15 | 196.75 | 195.33 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 221.25 | 227.12 | 227.24 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 229.30 | 226.89 | 226.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 230.75 | 228.04 | 227.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 10:15:00 | 228.00 | 228.03 | 227.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 11:15:00 | 226.88 | 227.80 | 227.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 11:15:00 | 226.88 | 227.80 | 227.32 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 14:15:00 | 225.95 | 227.06 | 227.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 09:15:00 | 224.30 | 226.44 | 226.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 219.95 | 219.81 | 221.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 218.53 | 218.98 | 220.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 218.53 | 218.98 | 220.48 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 14:15:00 | 222.95 | 220.99 | 220.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 09:15:00 | 226.00 | 222.42 | 221.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 230.80 | 232.98 | 229.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 15:15:00 | 230.10 | 232.40 | 229.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 15:15:00 | 230.10 | 232.40 | 229.74 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 15:15:00 | 228.10 | 231.56 | 231.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 227.53 | 230.76 | 231.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 14:15:00 | 227.23 | 226.93 | 228.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 233.43 | 228.32 | 228.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 233.43 | 228.32 | 228.73 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 234.10 | 229.48 | 229.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 11:15:00 | 236.45 | 230.87 | 229.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 238.55 | 239.14 | 236.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 238.55 | 239.14 | 236.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 238.55 | 239.14 | 236.90 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 15:15:00 | 234.38 | 236.26 | 236.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 228.20 | 234.65 | 235.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 10:15:00 | 232.03 | 231.99 | 233.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 12:15:00 | 232.50 | 232.16 | 233.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 12:15:00 | 232.50 | 232.16 | 233.16 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 10:15:00 | 237.50 | 234.25 | 233.84 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 229.50 | 233.83 | 234.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 227.43 | 232.55 | 233.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 228.40 | 228.09 | 229.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 231.75 | 228.68 | 229.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 231.75 | 228.68 | 229.72 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 232.90 | 229.54 | 229.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 237.13 | 231.06 | 230.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 237.38 | 237.51 | 234.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 11:15:00 | 234.15 | 236.86 | 235.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 234.15 | 236.86 | 235.50 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 15:15:00 | 251.00 | 251.32 | 251.36 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 252.25 | 251.51 | 251.44 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 250.68 | 251.34 | 251.37 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 12:15:00 | 252.35 | 251.49 | 251.43 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 250.93 | 251.38 | 251.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 250.10 | 251.12 | 251.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 15:15:00 | 247.98 | 247.33 | 248.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 240.15 | 245.90 | 247.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 240.15 | 245.90 | 247.94 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 242.48 | 240.30 | 240.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 244.98 | 241.51 | 240.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 14:15:00 | 243.78 | 243.83 | 242.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 10:15:00 | 258.70 | 259.55 | 257.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 258.70 | 259.55 | 257.25 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 250.68 | 255.88 | 256.12 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 256.48 | 255.41 | 255.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 14:15:00 | 259.35 | 256.56 | 255.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 10:15:00 | 254.53 | 256.65 | 256.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 10:15:00 | 254.53 | 256.65 | 256.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 254.53 | 256.65 | 256.15 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 15:15:00 | 254.73 | 256.72 | 256.74 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 260.68 | 257.51 | 257.09 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 251.98 | 255.96 | 256.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 250.50 | 254.09 | 255.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 251.75 | 250.54 | 252.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 252.48 | 250.92 | 252.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 252.48 | 250.92 | 252.48 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 260.95 | 253.58 | 253.46 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 255.75 | 259.96 | 260.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 251.95 | 257.55 | 258.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 259.27 | 256.96 | 257.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 259.27 | 256.96 | 257.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 259.27 | 256.96 | 257.94 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 255.83 | 240.70 | 239.85 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 238.40 | 241.27 | 241.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 236.35 | 239.85 | 240.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 239.65 | 236.74 | 237.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 239.65 | 236.74 | 237.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 239.65 | 236.74 | 237.92 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 242.25 | 238.74 | 238.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 246.23 | 240.82 | 239.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 11:15:00 | 247.40 | 248.51 | 246.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 13:15:00 | 247.35 | 248.51 | 246.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 247.35 | 248.51 | 246.86 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 254.65 | 255.41 | 255.50 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 11:15:00 | 256.63 | 255.65 | 255.61 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 254.00 | 255.30 | 255.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 14:15:00 | 253.65 | 254.76 | 255.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 15:15:00 | 255.03 | 254.81 | 255.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 15:15:00 | 255.03 | 254.81 | 255.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 255.03 | 254.81 | 255.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:30:00 | 252.63 | 254.50 | 254.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 10:45:00 | 252.00 | 254.05 | 254.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 240.00 | 249.96 | 252.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 239.40 | 249.96 | 252.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 247.78 | 247.53 | 249.55 | SL hit (close>ema200) qty=0.50 sl=247.53 alert=retest2 |

### Cycle 70 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 245.80 | 244.57 | 244.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 15:15:00 | 246.50 | 245.17 | 244.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 247.50 | 248.21 | 246.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:15:00 | 249.40 | 248.21 | 246.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 249.63 | 248.49 | 247.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:15:00 | 250.90 | 248.49 | 247.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 250.93 | 249.10 | 248.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:15:00 | 250.95 | 249.21 | 248.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 12:00:00 | 250.85 | 249.54 | 248.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 251.95 | 252.58 | 251.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:30:00 | 251.45 | 252.58 | 251.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 250.10 | 252.08 | 251.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 253.45 | 252.08 | 251.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:15:00 | 253.85 | 252.60 | 252.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 12:15:00 | 248.70 | 255.66 | 256.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 248.70 | 255.66 | 256.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 245.70 | 248.26 | 250.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 11:15:00 | 248.95 | 247.99 | 249.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 12:00:00 | 248.95 | 247.99 | 249.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 249.50 | 248.29 | 249.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:45:00 | 249.38 | 248.29 | 249.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 249.45 | 248.52 | 249.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:00:00 | 249.45 | 248.52 | 249.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 247.30 | 248.28 | 249.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:30:00 | 249.65 | 248.28 | 249.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 251.00 | 248.75 | 249.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:15:00 | 253.63 | 248.75 | 249.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 253.38 | 249.68 | 249.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:30:00 | 253.93 | 249.68 | 249.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 252.40 | 250.22 | 250.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 15:15:00 | 254.53 | 251.81 | 250.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 256.65 | 262.54 | 259.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 256.65 | 262.54 | 259.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 256.65 | 262.54 | 259.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 258.23 | 262.54 | 259.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 256.33 | 261.30 | 259.55 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 248.88 | 258.17 | 258.39 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 255.85 | 254.02 | 253.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 258.18 | 255.29 | 254.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 14:15:00 | 254.98 | 255.35 | 254.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 14:15:00 | 254.98 | 255.35 | 254.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 254.98 | 255.35 | 254.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:15:00 | 254.55 | 255.35 | 254.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 254.55 | 255.19 | 254.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 256.15 | 255.19 | 254.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 255.00 | 255.15 | 254.74 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 253.60 | 254.50 | 254.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 253.15 | 254.23 | 254.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 253.75 | 253.57 | 254.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 13:15:00 | 253.75 | 253.57 | 254.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 253.75 | 253.57 | 254.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:00:00 | 253.75 | 253.57 | 254.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 252.03 | 253.26 | 253.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:30:00 | 253.50 | 253.26 | 253.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 250.85 | 252.58 | 253.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 249.60 | 252.15 | 253.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:00:00 | 249.98 | 249.45 | 250.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:45:00 | 250.03 | 250.28 | 250.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:45:00 | 250.03 | 250.15 | 250.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 247.60 | 249.17 | 249.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:30:00 | 246.73 | 248.68 | 249.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 253.80 | 247.80 | 248.55 | SL hit (close>static) qty=1.00 sl=253.70 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 255.30 | 250.18 | 249.55 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 240.20 | 248.42 | 249.48 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 257.25 | 248.97 | 248.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 282.95 | 258.04 | 252.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 320.35 | 321.74 | 311.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 13:00:00 | 320.35 | 321.74 | 311.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 317.10 | 318.38 | 316.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:30:00 | 315.80 | 318.38 | 316.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 336.50 | 323.97 | 320.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:15:00 | 343.00 | 323.97 | 320.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-24 09:15:00 | 377.30 | 366.98 | 359.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 357.95 | 368.64 | 369.68 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 378.33 | 369.89 | 369.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 378.75 | 374.45 | 372.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 373.55 | 376.77 | 374.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 15:15:00 | 373.55 | 376.77 | 374.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 373.55 | 376.77 | 374.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:30:00 | 379.38 | 377.21 | 375.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 374.30 | 377.25 | 375.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 374.30 | 377.25 | 375.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 378.90 | 377.58 | 376.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:15:00 | 380.00 | 377.64 | 376.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 373.00 | 376.25 | 376.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 373.00 | 376.25 | 376.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 11:15:00 | 368.13 | 373.69 | 375.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 371.00 | 360.93 | 364.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 371.00 | 360.93 | 364.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 371.00 | 360.93 | 364.47 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 373.40 | 367.39 | 366.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 11:15:00 | 376.88 | 371.76 | 369.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 15:15:00 | 372.95 | 373.23 | 371.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 370.25 | 373.23 | 371.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 366.48 | 371.88 | 370.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 366.48 | 371.88 | 370.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 364.05 | 370.32 | 370.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 359.90 | 370.32 | 370.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 366.20 | 369.49 | 369.71 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 371.75 | 369.50 | 369.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 15:15:00 | 374.00 | 371.35 | 370.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 409.93 | 413.43 | 403.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 10:30:00 | 411.35 | 413.43 | 403.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 404.23 | 410.60 | 405.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 401.98 | 410.60 | 405.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 391.28 | 406.74 | 404.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 391.28 | 406.74 | 404.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 384.48 | 402.29 | 402.54 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 408.98 | 399.91 | 399.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 13:15:00 | 413.00 | 402.53 | 400.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 406.75 | 408.14 | 404.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 406.75 | 408.14 | 404.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 406.75 | 408.14 | 404.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 407.13 | 408.14 | 404.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 411.95 | 408.90 | 405.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 410.58 | 408.90 | 405.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 405.85 | 408.29 | 405.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 405.85 | 408.29 | 405.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 406.25 | 407.88 | 405.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 432.10 | 407.88 | 405.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 437.50 | 454.34 | 454.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 437.50 | 454.34 | 454.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 435.03 | 450.48 | 452.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 452.65 | 441.87 | 446.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 452.65 | 441.87 | 446.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 452.65 | 441.87 | 446.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 452.65 | 441.87 | 446.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 453.60 | 444.21 | 447.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 453.60 | 444.21 | 447.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 450.88 | 445.55 | 447.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:15:00 | 448.50 | 446.32 | 447.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 446.58 | 446.55 | 447.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 449.98 | 448.39 | 448.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 449.98 | 448.39 | 448.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 11:15:00 | 454.65 | 449.64 | 448.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 11:15:00 | 471.05 | 471.36 | 464.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 11:45:00 | 472.20 | 471.36 | 464.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 463.03 | 469.17 | 464.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 463.03 | 469.17 | 464.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 459.03 | 467.14 | 463.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 460.38 | 467.14 | 463.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 464.98 | 466.58 | 464.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 464.98 | 466.58 | 464.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 462.53 | 465.77 | 464.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:30:00 | 467.88 | 465.80 | 464.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 13:15:00 | 464.93 | 465.21 | 464.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 456.70 | 463.28 | 463.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 456.70 | 463.28 | 463.96 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 469.05 | 464.16 | 463.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 476.98 | 468.00 | 465.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 468.78 | 469.99 | 468.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 468.78 | 469.99 | 468.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 468.78 | 469.99 | 468.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 468.78 | 469.99 | 468.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 469.60 | 469.92 | 468.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 466.90 | 469.92 | 468.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 467.48 | 469.43 | 468.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 466.93 | 469.43 | 468.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 469.35 | 469.41 | 468.39 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 464.68 | 467.52 | 467.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 12:15:00 | 461.18 | 465.59 | 466.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 13:15:00 | 462.55 | 462.04 | 463.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:00:00 | 462.55 | 462.04 | 463.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 466.00 | 462.83 | 464.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 466.00 | 462.83 | 464.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 466.50 | 463.56 | 464.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 471.98 | 463.56 | 464.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 468.38 | 464.53 | 464.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:30:00 | 462.50 | 464.81 | 464.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 466.63 | 465.17 | 464.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 11:15:00 | 466.63 | 465.17 | 464.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 481.30 | 469.25 | 467.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 10:15:00 | 480.90 | 482.06 | 478.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 11:00:00 | 480.90 | 482.06 | 478.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 483.10 | 484.44 | 482.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 483.10 | 484.44 | 482.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 479.33 | 483.42 | 481.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:45:00 | 479.63 | 483.42 | 481.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 479.50 | 482.63 | 481.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 477.45 | 482.63 | 481.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 477.48 | 480.77 | 480.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 475.98 | 478.38 | 479.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 15:15:00 | 478.98 | 478.50 | 479.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 09:15:00 | 474.28 | 478.50 | 479.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 477.10 | 478.35 | 479.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:45:00 | 474.75 | 477.23 | 478.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:15:00 | 474.38 | 477.23 | 478.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 14:00:00 | 473.85 | 476.56 | 478.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 474.55 | 474.55 | 476.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 477.00 | 475.06 | 476.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-03 13:15:00 | 477.00 | 475.06 | 476.25 | SL hit (close>ema400) qty=1.00 sl=476.25 alert=retest1 |

### Cycle 94 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 492.35 | 479.37 | 477.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 496.90 | 488.11 | 484.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 488.78 | 489.38 | 485.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:45:00 | 488.68 | 489.38 | 485.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 488.60 | 489.33 | 486.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:00:00 | 488.60 | 489.33 | 486.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 484.58 | 488.52 | 486.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 485.45 | 488.52 | 486.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 489.50 | 488.72 | 486.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 497.18 | 490.41 | 488.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-13 14:15:00 | 546.90 | 530.66 | 518.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 593.30 | 607.53 | 609.21 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 634.42 | 612.10 | 609.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 636.67 | 620.61 | 613.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 611.20 | 624.40 | 618.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 611.20 | 624.40 | 618.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 611.20 | 624.40 | 618.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 611.20 | 624.40 | 618.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 600.92 | 619.70 | 616.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 600.92 | 619.70 | 616.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 597.50 | 612.78 | 614.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 590.50 | 608.32 | 611.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 588.45 | 583.19 | 590.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 588.45 | 583.19 | 590.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 586.00 | 583.75 | 589.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 600.75 | 583.75 | 589.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 593.08 | 585.62 | 590.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 591.00 | 585.62 | 590.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:30:00 | 589.42 | 587.48 | 589.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:15:00 | 561.45 | 575.04 | 581.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:15:00 | 559.95 | 575.04 | 581.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-03 09:15:00 | 531.90 | 553.88 | 565.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 533.48 | 519.80 | 519.17 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 517.98 | 521.26 | 521.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 11:15:00 | 509.40 | 517.70 | 519.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 514.65 | 514.60 | 517.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 513.50 | 513.65 | 515.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 513.50 | 513.65 | 515.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 513.50 | 513.65 | 515.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 511.83 | 513.34 | 515.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 515.00 | 513.34 | 515.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 513.17 | 511.51 | 513.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 513.17 | 511.51 | 513.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 511.50 | 511.51 | 513.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 509.08 | 511.51 | 513.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:15:00 | 483.63 | 500.55 | 504.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 458.17 | 469.62 | 479.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 100 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 467.18 | 456.91 | 456.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 477.20 | 468.77 | 463.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 473.28 | 473.33 | 470.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:15:00 | 474.35 | 473.33 | 470.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 497.45 | 500.76 | 496.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 496.98 | 500.76 | 496.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 495.53 | 499.71 | 496.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 495.53 | 499.71 | 496.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 496.23 | 499.01 | 496.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 493.85 | 499.01 | 496.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 494.95 | 498.20 | 496.46 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 483.00 | 493.59 | 494.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 477.83 | 490.44 | 493.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 12:15:00 | 475.40 | 471.47 | 478.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 12:15:00 | 475.40 | 471.47 | 478.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 475.40 | 471.47 | 478.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:45:00 | 477.50 | 471.47 | 478.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 483.48 | 472.99 | 477.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 472.23 | 472.93 | 476.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 448.62 | 455.71 | 463.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 453.20 | 451.25 | 458.79 | SL hit (close>ema200) qty=0.50 sl=451.25 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 470.08 | 460.07 | 459.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 483.75 | 468.38 | 463.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 471.68 | 474.03 | 468.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 471.68 | 474.03 | 468.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 475.00 | 474.22 | 469.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 468.70 | 474.22 | 469.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 466.38 | 472.65 | 469.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 469.93 | 472.65 | 469.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 466.68 | 471.46 | 468.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:45:00 | 469.45 | 471.00 | 468.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 473.35 | 470.62 | 469.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:45:00 | 470.80 | 470.07 | 469.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 469.80 | 470.07 | 469.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 469.08 | 469.87 | 469.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 468.98 | 469.87 | 469.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 469.00 | 469.69 | 469.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:15:00 | 468.83 | 469.69 | 469.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-22 12:15:00 | 466.00 | 468.96 | 469.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 466.00 | 468.96 | 469.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 13:15:00 | 465.28 | 468.22 | 468.69 | Break + close below crossover candle low |

### Cycle 104 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 487.00 | 471.03 | 469.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 500.20 | 476.87 | 472.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 532.58 | 540.46 | 527.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 532.58 | 540.46 | 527.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 530.63 | 538.50 | 527.82 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 511.48 | 524.86 | 525.36 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 524.98 | 522.02 | 521.62 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 09:15:00 | 515.55 | 521.14 | 521.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 15:15:00 | 513.50 | 517.03 | 518.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 09:15:00 | 527.50 | 519.12 | 519.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 527.50 | 519.12 | 519.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 527.50 | 519.12 | 519.64 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 523.75 | 520.05 | 520.01 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 519.00 | 520.15 | 520.28 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 521.50 | 520.40 | 520.37 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 519.73 | 520.26 | 520.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 508.03 | 517.69 | 519.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 526.73 | 518.93 | 519.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 11:15:00 | 526.73 | 518.93 | 519.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 526.73 | 518.93 | 519.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:30:00 | 531.78 | 518.93 | 519.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 12:15:00 | 523.17 | 519.78 | 519.75 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 516.03 | 519.39 | 519.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 511.00 | 515.89 | 517.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 516.92 | 516.09 | 517.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 516.92 | 516.09 | 517.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 513.00 | 515.65 | 517.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:45:00 | 507.85 | 511.26 | 513.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:00:00 | 507.38 | 510.49 | 513.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:30:00 | 507.20 | 510.09 | 512.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:00:00 | 507.65 | 509.60 | 512.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 512.45 | 504.53 | 506.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 512.73 | 504.53 | 506.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 508.48 | 505.32 | 506.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:30:00 | 508.13 | 505.29 | 506.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 501.20 | 506.64 | 506.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 508.23 | 505.91 | 506.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 14:15:00 | 511.53 | 507.04 | 506.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 14:15:00 | 511.53 | 507.04 | 506.77 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 506.00 | 507.10 | 507.22 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 10:15:00 | 508.78 | 507.42 | 507.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 15:15:00 | 509.83 | 508.11 | 507.76 | Break + close above crossover candle high |

### Cycle 117 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 504.08 | 507.30 | 507.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 11:15:00 | 501.45 | 505.43 | 506.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 496.03 | 494.84 | 498.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 496.03 | 494.84 | 498.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 496.03 | 494.84 | 498.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:30:00 | 498.13 | 494.84 | 498.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 497.98 | 495.47 | 498.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:45:00 | 493.53 | 495.49 | 497.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:30:00 | 492.53 | 494.82 | 496.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 14:15:00 | 493.00 | 494.67 | 496.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 493.63 | 494.81 | 495.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 494.38 | 494.72 | 495.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:45:00 | 495.00 | 494.72 | 495.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 496.60 | 495.10 | 495.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:45:00 | 496.98 | 495.10 | 495.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 497.10 | 495.50 | 495.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 497.10 | 495.50 | 495.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 14:15:00 | 498.30 | 496.06 | 496.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 498.30 | 496.06 | 496.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 15:15:00 | 501.48 | 497.14 | 496.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 494.50 | 496.61 | 496.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 494.50 | 496.61 | 496.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 494.50 | 496.61 | 496.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 494.50 | 496.61 | 496.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 495.13 | 496.32 | 496.25 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 493.13 | 495.68 | 495.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 491.50 | 493.81 | 494.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 495.18 | 494.08 | 494.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 495.18 | 494.08 | 494.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 495.18 | 494.08 | 494.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 495.90 | 494.08 | 494.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 493.60 | 493.99 | 494.75 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 503.25 | 496.46 | 495.64 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 492.05 | 497.69 | 498.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 490.00 | 496.15 | 497.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 470.70 | 468.06 | 477.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 470.70 | 468.06 | 477.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 476.93 | 470.38 | 476.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 476.93 | 470.38 | 476.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 479.35 | 472.17 | 477.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 479.35 | 472.17 | 477.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 477.75 | 473.29 | 477.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 477.75 | 473.29 | 477.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 480.00 | 474.63 | 477.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 470.50 | 474.63 | 477.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 470.93 | 473.67 | 476.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 467.80 | 472.17 | 475.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 465.78 | 470.16 | 473.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:30:00 | 467.50 | 469.31 | 472.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 15:15:00 | 444.41 | 451.88 | 459.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 15:15:00 | 444.12 | 451.88 | 459.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 442.49 | 449.22 | 457.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 421.02 | 435.28 | 446.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 450.50 | 436.25 | 435.53 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 431.85 | 438.74 | 439.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 428.75 | 433.27 | 435.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 422.60 | 417.68 | 423.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 422.60 | 417.68 | 423.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 422.60 | 417.68 | 423.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 422.60 | 417.68 | 423.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 432.80 | 421.16 | 424.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 432.80 | 421.16 | 424.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 434.40 | 423.81 | 425.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 434.90 | 423.81 | 425.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 441.00 | 428.90 | 427.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 444.35 | 431.99 | 428.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 431.00 | 433.87 | 430.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:00:00 | 431.00 | 433.87 | 430.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 439.05 | 434.90 | 431.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:15:00 | 442.00 | 435.91 | 432.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 13:00:00 | 441.65 | 437.06 | 432.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 411.10 | 431.40 | 431.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 411.10 | 431.40 | 431.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 396.50 | 413.74 | 421.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 423.25 | 410.04 | 414.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 423.25 | 410.04 | 414.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 423.25 | 410.04 | 414.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 423.25 | 410.04 | 414.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 421.95 | 412.42 | 415.19 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 426.00 | 417.26 | 417.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 429.00 | 419.61 | 418.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 436.75 | 438.44 | 430.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 436.75 | 438.44 | 430.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 434.85 | 436.91 | 431.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 436.45 | 436.91 | 431.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 427.95 | 435.11 | 430.83 | SL hit (close<static) qty=1.00 sl=430.30 alert=retest2 |

### Cycle 127 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 422.45 | 432.28 | 433.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 416.40 | 429.11 | 431.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 430.15 | 427.52 | 430.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 14:15:00 | 430.15 | 427.52 | 430.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 430.15 | 427.52 | 430.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 430.15 | 427.52 | 430.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 430.20 | 428.06 | 430.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 436.80 | 428.06 | 430.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 433.15 | 429.08 | 430.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:00:00 | 428.10 | 428.88 | 430.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:45:00 | 429.00 | 428.87 | 429.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 14:15:00 | 433.00 | 429.94 | 429.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 14:15:00 | 433.00 | 429.94 | 429.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 11:15:00 | 437.25 | 431.90 | 430.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 425.70 | 434.44 | 432.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 425.70 | 434.44 | 432.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 425.70 | 434.44 | 432.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 425.70 | 434.44 | 432.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 430.15 | 433.58 | 432.62 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 425.10 | 431.21 | 431.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 15:15:00 | 421.05 | 429.18 | 430.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 382.30 | 379.00 | 389.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 14:45:00 | 382.35 | 379.00 | 389.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 381.35 | 380.22 | 387.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 389.05 | 380.22 | 387.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 371.75 | 376.46 | 382.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 371.25 | 376.46 | 382.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 389.35 | 373.84 | 376.07 | SL hit (close>static) qty=1.00 sl=384.30 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 12:15:00 | 387.25 | 378.08 | 377.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 15:15:00 | 398.00 | 385.92 | 381.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 09:15:00 | 382.20 | 385.18 | 381.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 382.20 | 385.18 | 381.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 382.20 | 385.18 | 381.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:00:00 | 382.20 | 385.18 | 381.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 376.85 | 383.51 | 381.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 376.85 | 383.51 | 381.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 373.35 | 381.48 | 380.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:30:00 | 373.80 | 381.48 | 380.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 373.85 | 379.95 | 379.99 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 402.00 | 383.33 | 381.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 411.00 | 392.16 | 385.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 415.60 | 416.18 | 406.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 415.60 | 416.18 | 406.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 407.95 | 413.79 | 407.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 407.95 | 413.79 | 407.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 407.00 | 412.43 | 407.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 414.10 | 407.79 | 407.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 402.70 | 406.42 | 406.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 402.70 | 406.42 | 406.74 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 412.00 | 406.99 | 406.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 415.45 | 412.40 | 410.11 | Break + close above crossover candle high |

### Cycle 135 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 397.70 | 410.68 | 410.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 380.85 | 392.43 | 399.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 391.85 | 388.43 | 393.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 391.85 | 388.43 | 393.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 391.85 | 388.43 | 393.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 391.85 | 388.43 | 393.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 389.00 | 388.54 | 393.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 387.35 | 387.77 | 392.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 405.85 | 389.11 | 391.04 | SL hit (close>static) qty=1.00 sl=393.45 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 404.70 | 394.66 | 393.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 406.60 | 397.05 | 394.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 430.15 | 435.79 | 427.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 430.15 | 435.79 | 427.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 430.15 | 435.79 | 427.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 430.15 | 435.79 | 427.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 428.40 | 432.38 | 428.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 429.00 | 432.38 | 428.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 424.40 | 430.78 | 427.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 424.40 | 430.78 | 427.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 427.00 | 430.03 | 427.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 422.00 | 430.03 | 427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 420.70 | 428.16 | 427.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:00:00 | 424.45 | 427.42 | 426.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 421.00 | 426.13 | 426.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 421.00 | 426.13 | 426.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 14:15:00 | 419.05 | 423.22 | 424.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 414.05 | 412.72 | 416.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:45:00 | 413.30 | 412.72 | 416.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 411.15 | 412.40 | 416.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:15:00 | 408.60 | 411.55 | 415.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 409.00 | 410.36 | 412.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 420.70 | 412.92 | 413.02 | SL hit (close>static) qty=1.00 sl=417.30 alert=retest2 |

### Cycle 138 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 422.25 | 414.79 | 413.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 425.55 | 420.22 | 417.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 15:15:00 | 482.35 | 484.00 | 474.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:15:00 | 480.05 | 484.00 | 474.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 473.45 | 481.89 | 474.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 473.45 | 481.89 | 474.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 476.45 | 480.80 | 474.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 472.65 | 480.80 | 474.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 472.55 | 479.15 | 474.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 470.70 | 479.15 | 474.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 470.80 | 477.48 | 473.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 469.00 | 477.48 | 473.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 470.20 | 475.58 | 473.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 470.20 | 475.58 | 473.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 469.00 | 474.26 | 473.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 476.45 | 474.26 | 473.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 483.10 | 485.40 | 480.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 492.45 | 485.41 | 481.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 496.70 | 487.34 | 483.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-01 09:15:00 | 541.70 | 515.57 | 504.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 496.55 | 519.02 | 519.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 493.45 | 508.03 | 513.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 475.15 | 464.94 | 477.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 13:00:00 | 475.15 | 464.94 | 477.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 453.35 | 463.71 | 472.87 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 487.95 | 470.95 | 470.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 489.00 | 483.37 | 478.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 484.50 | 487.23 | 482.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 484.50 | 487.23 | 482.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 484.50 | 487.23 | 482.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:15:00 | 479.00 | 487.23 | 482.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 482.80 | 486.35 | 482.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:30:00 | 481.35 | 486.35 | 482.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 485.45 | 486.17 | 482.84 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 09:15:00 | 471.30 | 481.66 | 481.87 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 484.90 | 479.06 | 478.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 488.60 | 483.01 | 481.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 485.50 | 486.31 | 483.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 485.50 | 486.31 | 483.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 477.10 | 488.24 | 487.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 477.10 | 488.24 | 487.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 10:15:00 | 473.40 | 485.27 | 485.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 13:15:00 | 470.55 | 478.63 | 482.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 09:15:00 | 479.00 | 476.49 | 480.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 479.00 | 476.49 | 480.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 479.00 | 476.49 | 480.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:30:00 | 486.10 | 476.49 | 480.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 471.35 | 475.46 | 479.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 11:15:00 | 467.25 | 475.46 | 479.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 466.00 | 474.23 | 475.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:00:00 | 467.70 | 471.57 | 473.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:00:00 | 459.00 | 469.06 | 472.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 459.60 | 462.36 | 467.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:15:00 | 446.50 | 456.85 | 459.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 12:15:00 | 443.89 | 454.48 | 458.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 12:15:00 | 444.31 | 454.48 | 458.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:00:00 | 445.00 | 454.48 | 458.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 13:15:00 | 442.70 | 451.81 | 456.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 436.05 | 448.37 | 454.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 10:15:00 | 444.75 | 443.85 | 450.82 | SL hit (close>ema200) qty=0.50 sl=443.85 alert=retest2 |

### Cycle 144 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 459.40 | 440.19 | 439.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 462.65 | 449.60 | 444.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 466.40 | 466.90 | 460.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:45:00 | 466.10 | 466.90 | 460.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 471.95 | 474.81 | 470.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 471.95 | 474.81 | 470.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 460.00 | 471.73 | 470.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:45:00 | 454.60 | 471.73 | 470.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 463.85 | 470.15 | 469.60 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 462.60 | 468.64 | 468.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 12:15:00 | 461.90 | 467.29 | 468.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 462.80 | 458.23 | 461.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 462.80 | 458.23 | 461.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 462.80 | 458.23 | 461.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 462.80 | 458.23 | 461.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 460.65 | 458.71 | 461.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 456.05 | 458.18 | 460.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 458.45 | 460.43 | 460.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 458.80 | 460.57 | 460.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 455.30 | 459.89 | 460.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 459.80 | 457.94 | 458.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 455.85 | 457.65 | 458.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 453.85 | 457.65 | 458.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 455.60 | 457.52 | 458.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 454.55 | 457.32 | 458.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 456.60 | 455.52 | 456.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 457.25 | 455.52 | 456.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 453.95 | 455.21 | 456.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 457.00 | 455.21 | 456.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 455.15 | 454.62 | 455.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 454.00 | 454.57 | 455.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 459.40 | 455.51 | 455.71 | SL hit (close>static) qty=1.00 sl=458.45 alert=retest2 |

### Cycle 146 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 475.00 | 460.05 | 457.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 12:15:00 | 489.20 | 489.47 | 481.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:15:00 | 488.35 | 489.47 | 481.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 483.00 | 487.75 | 482.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 483.85 | 487.75 | 482.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 485.15 | 487.23 | 483.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:15:00 | 491.10 | 487.23 | 483.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 10:15:00 | 540.21 | 507.22 | 495.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 520.95 | 524.83 | 524.87 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 533.40 | 526.54 | 525.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 570.60 | 537.38 | 531.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 552.55 | 555.24 | 546.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 552.55 | 555.24 | 546.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 552.55 | 555.24 | 546.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 591.05 | 552.66 | 548.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 562.05 | 564.40 | 562.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 559.95 | 563.62 | 562.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 560.50 | 563.62 | 562.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 554.00 | 560.07 | 561.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 562.50 | 558.28 | 559.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 562.50 | 558.28 | 559.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 562.50 | 558.28 | 559.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 568.60 | 558.28 | 559.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 555.80 | 557.78 | 559.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 554.10 | 556.75 | 558.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 563.35 | 552.49 | 555.23 | SL hit (close>static) qty=1.00 sl=562.70 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 567.50 | 557.94 | 557.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 581.80 | 567.15 | 562.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 580.70 | 582.68 | 576.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 12:30:00 | 580.45 | 582.68 | 576.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 576.90 | 581.52 | 576.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 577.40 | 581.52 | 576.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 571.60 | 579.54 | 575.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 571.60 | 579.54 | 575.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 571.00 | 577.83 | 575.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 589.10 | 577.83 | 575.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 592.50 | 595.73 | 595.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 592.50 | 595.73 | 595.99 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 601.75 | 596.03 | 595.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 610.80 | 602.30 | 599.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 609.70 | 609.76 | 605.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 609.70 | 609.76 | 605.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 602.30 | 608.58 | 606.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 602.30 | 608.58 | 606.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 603.45 | 607.55 | 606.27 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 596.70 | 603.94 | 604.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 579.30 | 596.14 | 600.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 589.40 | 588.91 | 594.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 608.25 | 588.91 | 594.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 596.60 | 590.45 | 594.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 603.45 | 590.45 | 594.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 600.35 | 592.43 | 594.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 600.35 | 592.43 | 594.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 594.50 | 595.49 | 595.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 593.80 | 595.49 | 595.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 593.00 | 594.99 | 595.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 590.00 | 594.08 | 595.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 589.50 | 593.16 | 594.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 590.30 | 592.92 | 593.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 590.00 | 591.88 | 593.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 592.10 | 590.84 | 592.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 592.10 | 590.84 | 592.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 589.75 | 590.62 | 591.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 589.25 | 590.62 | 591.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 593.10 | 591.11 | 592.07 | SL hit (close>static) qty=1.00 sl=592.20 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 612.60 | 599.63 | 595.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 613.25 | 613.61 | 609.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 13:00:00 | 613.25 | 613.61 | 609.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 608.40 | 613.48 | 610.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 608.40 | 613.48 | 610.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 609.95 | 612.77 | 610.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 605.75 | 612.77 | 610.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 608.00 | 611.82 | 610.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 607.95 | 611.82 | 610.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 605.95 | 610.64 | 609.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 605.75 | 610.64 | 609.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 608.50 | 609.44 | 609.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 603.35 | 608.22 | 608.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 607.70 | 607.52 | 608.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 607.70 | 607.52 | 608.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 607.70 | 607.52 | 608.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 609.30 | 607.52 | 608.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 606.85 | 607.08 | 608.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 608.15 | 607.08 | 608.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 605.15 | 606.70 | 607.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 613.20 | 606.70 | 607.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 613.50 | 608.06 | 608.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 616.80 | 608.06 | 608.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 610.60 | 608.57 | 608.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 616.85 | 610.50 | 609.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 610.10 | 611.10 | 609.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 610.10 | 611.10 | 609.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 633.50 | 615.42 | 612.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 637.90 | 628.80 | 621.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 616.35 | 623.84 | 624.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 616.35 | 623.84 | 624.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 610.60 | 621.19 | 622.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 601.35 | 600.46 | 606.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 601.35 | 600.46 | 606.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 605.95 | 601.75 | 606.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 612.00 | 601.75 | 606.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 605.80 | 602.56 | 606.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 604.25 | 602.56 | 606.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 611.60 | 608.24 | 607.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 611.60 | 608.24 | 607.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 615.60 | 609.98 | 608.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 624.90 | 628.89 | 623.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 624.90 | 628.89 | 623.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 617.00 | 626.51 | 622.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 624.45 | 626.51 | 622.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 620.75 | 625.36 | 622.54 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 619.70 | 621.20 | 621.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 615.50 | 620.06 | 620.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 606.95 | 603.66 | 607.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 606.95 | 603.66 | 607.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 606.95 | 603.66 | 607.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 606.95 | 603.66 | 607.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 608.10 | 604.55 | 607.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 607.60 | 604.55 | 607.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 605.20 | 604.68 | 607.61 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 621.60 | 609.32 | 608.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 626.15 | 617.01 | 613.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 619.05 | 619.49 | 615.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:45:00 | 619.00 | 619.49 | 615.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 618.10 | 619.21 | 615.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 618.10 | 619.21 | 615.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 616.20 | 619.99 | 617.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 615.30 | 619.99 | 617.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 616.65 | 619.32 | 617.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 617.15 | 619.32 | 617.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 616.05 | 618.67 | 617.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:30:00 | 614.45 | 618.67 | 617.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 616.30 | 618.31 | 617.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 616.30 | 618.31 | 617.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 615.65 | 617.78 | 617.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 615.65 | 617.78 | 617.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 614.50 | 617.12 | 617.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 618.70 | 617.12 | 617.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 582.20 | 610.80 | 614.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 582.20 | 610.80 | 614.24 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 596.30 | 592.39 | 591.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 605.65 | 596.79 | 594.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 669.90 | 671.20 | 656.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 669.90 | 671.20 | 656.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 658.40 | 665.60 | 659.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 658.40 | 665.60 | 659.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 665.00 | 665.48 | 659.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 667.10 | 664.40 | 659.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 673.85 | 665.78 | 661.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 657.00 | 662.82 | 661.43 | SL hit (close<static) qty=1.00 sl=657.85 alert=retest2 |

### Cycle 163 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 693.70 | 698.72 | 698.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 679.55 | 693.94 | 696.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 687.65 | 684.92 | 689.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 687.65 | 684.92 | 689.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 687.65 | 684.92 | 689.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 687.65 | 684.92 | 689.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 690.40 | 686.02 | 689.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 690.40 | 686.02 | 689.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 688.65 | 686.54 | 689.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 688.65 | 686.54 | 689.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 691.00 | 687.43 | 689.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 690.10 | 687.43 | 689.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 688.35 | 687.62 | 689.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 685.95 | 687.28 | 688.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 683.30 | 686.42 | 688.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:15:00 | 685.70 | 687.18 | 688.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 702.80 | 690.30 | 689.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 702.80 | 690.30 | 689.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 704.20 | 701.00 | 698.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 703.15 | 703.96 | 700.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 703.15 | 703.96 | 700.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 703.20 | 704.70 | 702.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 712.70 | 704.70 | 702.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 707.70 | 709.19 | 705.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 709.00 | 707.68 | 705.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 712.25 | 706.94 | 705.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 714.05 | 708.36 | 706.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 715.40 | 709.74 | 709.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 718.60 | 714.15 | 712.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 13:15:00 | 715.20 | 716.63 | 713.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 715.20 | 716.63 | 713.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 717.00 | 716.71 | 714.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 724.40 | 717.49 | 714.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 709.35 | 715.71 | 715.18 | SL hit (close<static) qty=1.00 sl=714.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 712.65 | 714.47 | 714.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 705.85 | 710.71 | 712.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 710.00 | 707.42 | 709.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 710.00 | 707.42 | 709.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 710.00 | 707.42 | 709.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 711.25 | 707.42 | 709.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 708.50 | 707.63 | 709.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 708.80 | 707.63 | 709.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 695.25 | 701.42 | 705.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 689.30 | 699.51 | 704.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:15:00 | 654.83 | 667.23 | 679.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 647.90 | 646.57 | 656.31 | SL hit (close>ema200) qty=0.50 sl=646.57 alert=retest2 |

### Cycle 168 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 613.25 | 611.29 | 611.11 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 606.75 | 610.55 | 610.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 602.55 | 608.54 | 609.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 605.50 | 605.01 | 607.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 605.50 | 605.01 | 607.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 606.50 | 605.15 | 606.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 606.50 | 605.15 | 606.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 604.70 | 605.06 | 606.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:15:00 | 613.50 | 605.06 | 606.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 611.40 | 606.32 | 607.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 618.50 | 606.32 | 607.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 613.90 | 607.84 | 607.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 616.00 | 611.31 | 609.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 15:15:00 | 615.55 | 615.79 | 613.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 611.30 | 615.79 | 613.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 609.30 | 614.49 | 612.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 609.30 | 614.49 | 612.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 613.10 | 614.21 | 612.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 611.80 | 614.21 | 612.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 611.45 | 613.66 | 612.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 610.60 | 613.66 | 612.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 616.00 | 614.13 | 612.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 625.15 | 616.30 | 614.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 612.15 | 617.70 | 618.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 612.15 | 617.70 | 618.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 604.55 | 611.19 | 614.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 14:15:00 | 608.00 | 607.29 | 610.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 15:00:00 | 608.00 | 607.29 | 610.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 595.95 | 605.15 | 609.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 594.20 | 603.21 | 608.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:00:00 | 595.45 | 603.21 | 608.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 595.20 | 599.19 | 604.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 564.49 | 574.26 | 587.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 565.68 | 574.26 | 587.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 565.44 | 574.26 | 587.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-11 10:15:00 | 534.78 | 549.85 | 565.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 172 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 552.30 | 549.56 | 549.43 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 547.05 | 549.58 | 549.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 546.30 | 548.69 | 549.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 541.95 | 541.41 | 544.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:45:00 | 542.35 | 541.41 | 544.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 543.10 | 541.75 | 544.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 548.00 | 541.75 | 544.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 542.05 | 541.81 | 543.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:45:00 | 543.00 | 541.81 | 543.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 543.10 | 542.07 | 543.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 544.25 | 542.07 | 543.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 544.25 | 542.50 | 543.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 544.25 | 542.50 | 543.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 544.90 | 542.98 | 544.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 536.50 | 542.98 | 544.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 509.67 | 516.95 | 526.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 514.45 | 513.42 | 519.96 | SL hit (close>ema200) qty=0.50 sl=513.42 alert=retest2 |

### Cycle 174 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 526.90 | 521.44 | 521.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 528.60 | 522.87 | 521.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 526.10 | 526.26 | 524.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 15:15:00 | 526.10 | 526.26 | 524.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 526.10 | 526.26 | 524.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 522.25 | 526.26 | 524.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 522.70 | 525.55 | 524.54 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 519.80 | 523.27 | 523.65 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 532.75 | 524.67 | 524.06 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 520.75 | 523.72 | 523.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 518.30 | 522.24 | 523.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 521.85 | 521.40 | 522.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 521.85 | 521.40 | 522.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 521.85 | 521.40 | 522.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:15:00 | 523.85 | 521.40 | 522.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 523.10 | 521.74 | 522.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 523.80 | 521.74 | 522.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 521.60 | 521.71 | 522.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 520.25 | 521.20 | 522.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 525.55 | 521.16 | 521.72 | SL hit (close>static) qty=1.00 sl=523.15 alert=retest2 |

### Cycle 178 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 527.50 | 522.40 | 522.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 528.00 | 524.97 | 523.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 12:15:00 | 528.00 | 529.82 | 527.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:00:00 | 528.00 | 529.82 | 527.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 526.45 | 529.14 | 527.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 526.00 | 529.14 | 527.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 525.90 | 528.49 | 527.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 525.90 | 528.49 | 527.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 527.25 | 528.24 | 527.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 521.50 | 528.24 | 527.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 518.05 | 526.21 | 526.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 516.95 | 522.13 | 524.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 525.50 | 520.34 | 522.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 525.50 | 520.34 | 522.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 525.50 | 520.34 | 522.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 525.50 | 520.34 | 522.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 537.75 | 523.82 | 523.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 537.75 | 523.82 | 523.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 540.65 | 527.19 | 525.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 548.60 | 531.47 | 527.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 562.15 | 562.55 | 556.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 13:30:00 | 560.95 | 562.55 | 556.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 562.50 | 567.35 | 563.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 560.95 | 567.35 | 563.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 568.70 | 567.62 | 564.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 572.30 | 567.81 | 564.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:00:00 | 571.55 | 568.56 | 565.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:45:00 | 571.30 | 569.24 | 565.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:30:00 | 571.50 | 568.96 | 565.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 567.00 | 568.89 | 566.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 570.90 | 568.07 | 567.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 573.20 | 567.55 | 567.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 573.65 | 573.37 | 572.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 575.00 | 571.78 | 571.76 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 566.40 | 570.71 | 571.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 563.60 | 568.41 | 570.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 561.00 | 555.37 | 557.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 561.00 | 555.37 | 557.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 561.00 | 555.37 | 557.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 561.55 | 555.37 | 557.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 565.80 | 557.46 | 558.56 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 567.15 | 560.20 | 559.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 573.00 | 562.27 | 560.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 567.00 | 567.15 | 564.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 567.00 | 567.15 | 564.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 567.00 | 567.15 | 564.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 564.55 | 566.65 | 564.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 566.80 | 566.68 | 564.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 568.90 | 567.19 | 565.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 568.05 | 570.59 | 568.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:00:00 | 569.85 | 570.59 | 568.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 571.95 | 570.84 | 568.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-06 13:15:00 | 625.79 | 596.34 | 584.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 583.90 | 593.60 | 594.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 15:15:00 | 579.60 | 585.22 | 589.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 569.50 | 565.61 | 571.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 569.50 | 565.61 | 571.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 569.50 | 565.61 | 571.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 569.45 | 565.61 | 571.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 571.00 | 566.69 | 571.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 571.00 | 566.69 | 571.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 571.00 | 567.55 | 571.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 571.25 | 567.55 | 571.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 570.75 | 568.19 | 571.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 569.50 | 568.19 | 571.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 566.80 | 567.91 | 570.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 565.70 | 568.74 | 570.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 565.55 | 568.74 | 570.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 565.00 | 568.20 | 569.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 565.80 | 567.86 | 569.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 555.95 | 555.69 | 560.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 551.30 | 555.69 | 560.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 551.20 | 553.91 | 559.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 550.50 | 553.23 | 558.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 537.41 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 537.27 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 536.75 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 537.51 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 553.95 | 539.79 | 544.78 | SL hit (close>ema200) qty=0.50 sl=539.79 alert=retest2 |

### Cycle 186 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 549.90 | 547.39 | 547.23 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 545.45 | 546.93 | 547.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 538.95 | 545.34 | 546.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 535.05 | 533.16 | 537.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 535.05 | 533.16 | 537.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 535.05 | 533.16 | 537.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 552.30 | 533.16 | 537.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 552.05 | 536.94 | 538.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 554.50 | 536.94 | 538.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 554.35 | 540.42 | 540.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 558.35 | 549.16 | 544.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 548.15 | 550.64 | 546.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 548.15 | 550.64 | 546.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 548.15 | 550.64 | 546.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 547.00 | 550.64 | 546.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 546.50 | 549.81 | 546.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 546.50 | 549.81 | 546.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 548.20 | 549.49 | 546.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 549.50 | 549.35 | 547.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:15:00 | 549.50 | 549.35 | 547.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 542.65 | 556.21 | 555.05 | SL hit (close<static) qty=1.00 sl=545.95 alert=retest2 |

### Cycle 189 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 542.25 | 553.42 | 553.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 540.20 | 550.78 | 552.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 543.30 | 542.07 | 546.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 543.30 | 542.07 | 546.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 545.65 | 542.79 | 546.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 546.45 | 542.79 | 546.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 550.90 | 544.41 | 547.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 550.90 | 544.41 | 547.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 546.20 | 544.77 | 546.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 572.50 | 544.77 | 546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 565.75 | 548.96 | 548.67 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 558.95 | 564.17 | 564.76 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 575.55 | 563.99 | 563.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 582.85 | 572.06 | 567.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 576.50 | 576.56 | 572.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:30:00 | 576.40 | 576.56 | 572.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 573.10 | 575.59 | 572.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 573.10 | 575.59 | 572.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 574.00 | 575.27 | 572.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:15:00 | 571.65 | 575.27 | 572.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 571.85 | 574.59 | 572.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:15:00 | 571.00 | 574.59 | 572.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 571.00 | 573.87 | 572.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 574.75 | 573.80 | 572.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 567.20 | 572.48 | 572.03 | SL hit (close<static) qty=1.00 sl=569.00 alert=retest2 |

### Cycle 193 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 567.90 | 571.56 | 571.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 557.40 | 566.00 | 568.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 553.90 | 553.84 | 558.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 553.90 | 553.84 | 558.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 550.10 | 553.16 | 557.49 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 568.50 | 559.66 | 558.56 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 554.30 | 558.71 | 558.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 553.80 | 557.33 | 558.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 563.55 | 556.88 | 557.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 563.55 | 556.88 | 557.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 563.55 | 556.88 | 557.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 567.50 | 556.88 | 557.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 564.05 | 558.31 | 558.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 575.50 | 568.68 | 565.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 593.85 | 594.29 | 587.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 593.85 | 594.29 | 587.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 576.90 | 590.79 | 588.60 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 576.70 | 585.24 | 586.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 568.55 | 578.32 | 582.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 566.90 | 564.83 | 571.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 563.50 | 564.56 | 570.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:45:00 | 562.75 | 564.54 | 570.33 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 566.00 | 564.83 | 569.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 568.00 | 564.83 | 569.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 572.05 | 566.62 | 569.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 572.05 | 566.62 | 569.89 | SL hit (close>ema400) qty=1.00 sl=569.89 alert=retest1 |

### Cycle 198 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 569.00 | 560.27 | 559.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 580.65 | 564.35 | 561.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 566.85 | 569.93 | 565.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 566.85 | 569.93 | 565.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 566.85 | 569.93 | 565.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 566.85 | 569.93 | 565.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 572.55 | 569.91 | 566.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 577.60 | 571.93 | 567.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 576.55 | 575.90 | 571.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 556.30 | 569.07 | 569.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 556.30 | 569.07 | 569.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 553.55 | 565.96 | 567.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 551.40 | 545.35 | 551.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 551.40 | 545.35 | 551.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 551.40 | 545.35 | 551.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 551.40 | 545.35 | 551.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 549.85 | 546.25 | 551.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 546.35 | 546.25 | 551.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 561.50 | 551.00 | 551.66 | SL hit (close>static) qty=1.00 sl=553.60 alert=retest2 |

### Cycle 200 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 563.85 | 553.57 | 552.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 564.80 | 557.28 | 554.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 554.25 | 559.68 | 557.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 554.25 | 559.68 | 557.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 554.25 | 559.68 | 557.01 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 544.20 | 553.70 | 554.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 539.50 | 550.49 | 552.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 551.10 | 547.40 | 550.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 13:15:00 | 551.10 | 547.40 | 550.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 551.10 | 547.40 | 550.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 14:00:00 | 551.10 | 547.40 | 550.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 544.80 | 546.88 | 549.70 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 559.50 | 551.00 | 550.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 563.75 | 553.55 | 551.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 568.10 | 568.90 | 562.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 563.55 | 568.90 | 562.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 558.40 | 566.80 | 562.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 558.40 | 566.80 | 562.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 557.25 | 564.89 | 562.02 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 553.80 | 560.29 | 560.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 551.00 | 558.43 | 559.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 553.55 | 543.21 | 548.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 553.55 | 543.21 | 548.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 553.55 | 543.21 | 548.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 554.00 | 543.21 | 548.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 551.30 | 544.83 | 548.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 550.30 | 544.83 | 548.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 554.10 | 551.31 | 551.14 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 545.50 | 550.46 | 550.91 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 554.85 | 551.56 | 551.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 559.80 | 553.20 | 552.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 552.50 | 554.72 | 553.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 552.50 | 554.72 | 553.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 552.50 | 554.72 | 553.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 552.50 | 554.72 | 553.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 557.85 | 555.34 | 553.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 563.85 | 556.84 | 554.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 562.10 | 557.71 | 555.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:45:00 | 564.00 | 559.56 | 556.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:45:00 | 564.80 | 561.37 | 557.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 562.00 | 562.79 | 560.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 573.35 | 562.79 | 560.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 620.24 | 610.97 | 602.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 672.10 | 689.68 | 691.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 662.00 | 684.14 | 689.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 668.00 | 665.93 | 675.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 668.55 | 665.93 | 675.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 665.60 | 666.40 | 671.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:45:00 | 665.45 | 666.40 | 671.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 673.60 | 667.84 | 671.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 673.60 | 667.84 | 671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 668.75 | 668.02 | 671.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 667.55 | 667.93 | 670.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 668.00 | 669.08 | 670.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 668.35 | 669.42 | 670.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 667.50 | 668.57 | 669.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 659.15 | 666.44 | 668.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 666.60 | 666.44 | 668.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 664.20 | 662.60 | 665.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:15:00 | 672.95 | 662.60 | 665.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | SL hit (close>static) qty=1.00 sl=674.30 alert=retest2 |

### Cycle 208 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 676.15 | 667.39 | 666.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 688.85 | 678.65 | 674.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 719.50 | 719.98 | 709.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 719.50 | 719.98 | 709.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:30:00 | 252.63 | 2024-04-15 09:15:00 | 240.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 10:45:00 | 252.00 | 2024-04-15 09:15:00 | 239.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 09:30:00 | 252.63 | 2024-04-16 09:15:00 | 247.78 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2024-04-12 10:45:00 | 252.00 | 2024-04-16 09:15:00 | 247.78 | STOP_HIT | 0.50 | 1.67% |
| BUY | retest2 | 2024-04-25 10:15:00 | 250.90 | 2024-05-07 12:15:00 | 248.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-04-26 09:15:00 | 250.93 | 2024-05-07 12:15:00 | 248.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-04-26 11:15:00 | 250.95 | 2024-05-07 12:15:00 | 248.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-04-26 12:00:00 | 250.85 | 2024-05-07 12:15:00 | 248.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-05-02 09:15:00 | 253.45 | 2024-05-07 12:15:00 | 248.70 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-05-02 12:15:00 | 253.85 | 2024-05-07 12:15:00 | 248.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-05-28 11:15:00 | 249.60 | 2024-06-03 09:15:00 | 253.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-05-29 15:00:00 | 249.98 | 2024-06-03 09:15:00 | 253.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-05-30 11:45:00 | 250.03 | 2024-06-03 09:15:00 | 253.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-05-30 13:45:00 | 250.03 | 2024-06-03 09:15:00 | 253.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-05-31 10:30:00 | 246.73 | 2024-06-03 09:15:00 | 253.80 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-06-18 10:15:00 | 343.00 | 2024-06-24 09:15:00 | 377.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 15:15:00 | 380.00 | 2024-07-04 09:15:00 | 373.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-24 09:15:00 | 432.10 | 2024-08-05 10:15:00 | 437.50 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2024-08-06 13:15:00 | 448.50 | 2024-08-07 10:15:00 | 449.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-08-06 14:30:00 | 446.58 | 2024-08-07 10:15:00 | 449.98 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-08-13 10:30:00 | 467.88 | 2024-08-13 14:15:00 | 456.70 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-08-13 13:15:00 | 464.93 | 2024-08-13 14:15:00 | 456.70 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-08-23 10:30:00 | 462.50 | 2024-08-23 11:15:00 | 466.63 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest1 | 2024-09-02 09:15:00 | 474.28 | 2024-09-03 13:15:00 | 477.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-09-02 12:45:00 | 474.75 | 2024-09-04 09:15:00 | 492.35 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-09-02 13:15:00 | 474.38 | 2024-09-04 09:15:00 | 492.35 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2024-09-02 14:00:00 | 473.85 | 2024-09-04 09:15:00 | 492.35 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2024-09-03 10:00:00 | 474.55 | 2024-09-04 09:15:00 | 492.35 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2024-09-10 09:15:00 | 497.18 | 2024-09-13 14:15:00 | 546.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-27 10:15:00 | 591.00 | 2024-10-01 09:15:00 | 561.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:30:00 | 589.42 | 2024-10-01 09:15:00 | 559.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 10:15:00 | 591.00 | 2024-10-03 09:15:00 | 531.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 14:30:00 | 589.42 | 2024-10-03 09:15:00 | 530.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 509.08 | 2024-10-21 09:15:00 | 483.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 509.08 | 2024-10-23 09:15:00 | 458.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 472.23 | 2024-11-13 14:15:00 | 448.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 472.23 | 2024-11-14 11:15:00 | 453.20 | STOP_HIT | 0.50 | 4.03% |
| BUY | retest2 | 2024-11-21 11:45:00 | 469.45 | 2024-11-22 12:15:00 | 466.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-11-22 09:15:00 | 473.35 | 2024-11-22 12:15:00 | 466.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-22 09:45:00 | 470.80 | 2024-11-22 12:15:00 | 466.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-11-22 10:15:00 | 469.80 | 2024-11-22 12:15:00 | 466.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-12 10:45:00 | 507.85 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-12-12 12:00:00 | 507.38 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-12-12 12:30:00 | 507.20 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-12 14:00:00 | 507.65 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-16 11:30:00 | 508.13 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-17 09:15:00 | 501.20 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-12-17 14:15:00 | 508.23 | 2024-12-17 14:15:00 | 511.53 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-24 14:45:00 | 493.53 | 2024-12-27 14:15:00 | 498.30 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-12-26 12:30:00 | 492.53 | 2024-12-27 14:15:00 | 498.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-12-26 14:15:00 | 493.00 | 2024-12-27 14:15:00 | 498.30 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-12-27 11:15:00 | 493.63 | 2024-12-27 14:15:00 | 498.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-01-08 11:30:00 | 467.80 | 2025-01-10 15:15:00 | 444.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 465.78 | 2025-01-10 15:15:00 | 444.12 | PARTIAL | 0.50 | 4.65% |
| SELL | retest2 | 2025-01-09 10:30:00 | 467.50 | 2025-01-13 09:15:00 | 442.49 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-01-08 11:30:00 | 467.80 | 2025-01-13 14:15:00 | 421.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 465.78 | 2025-01-13 14:15:00 | 420.75 | TARGET_HIT | 0.50 | 9.67% |
| SELL | retest2 | 2025-01-09 10:30:00 | 467.50 | 2025-01-13 15:15:00 | 419.20 | TARGET_HIT | 0.50 | 10.33% |
| BUY | retest2 | 2025-01-24 12:15:00 | 442.00 | 2025-01-27 09:15:00 | 411.10 | STOP_HIT | 1.00 | -6.99% |
| BUY | retest2 | 2025-01-24 13:00:00 | 441.65 | 2025-01-27 09:15:00 | 411.10 | STOP_HIT | 1.00 | -6.92% |
| BUY | retest2 | 2025-01-31 09:15:00 | 436.45 | 2025-01-31 09:15:00 | 427.95 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-31 10:30:00 | 437.10 | 2025-01-31 11:15:00 | 427.35 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-01-31 14:15:00 | 435.10 | 2025-02-03 09:15:00 | 427.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-02-03 09:15:00 | 437.95 | 2025-02-03 09:15:00 | 427.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-04 11:00:00 | 428.10 | 2025-02-05 14:15:00 | 433.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-02-05 09:45:00 | 429.00 | 2025-02-05 14:15:00 | 433.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-02-14 10:15:00 | 371.25 | 2025-02-17 10:15:00 | 389.35 | STOP_HIT | 1.00 | -4.88% |
| BUY | retest2 | 2025-02-24 11:30:00 | 414.10 | 2025-02-24 14:15:00 | 402.70 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-03-04 11:30:00 | 387.35 | 2025-03-05 09:15:00 | 405.85 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2025-03-11 11:00:00 | 424.45 | 2025-03-11 11:15:00 | 421.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-03-13 13:15:00 | 408.60 | 2025-03-18 09:15:00 | 420.70 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-03-17 14:15:00 | 409.00 | 2025-03-18 09:15:00 | 420.70 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-03-27 10:30:00 | 492.45 | 2025-04-01 09:15:00 | 541.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-27 15:00:00 | 496.70 | 2025-04-01 09:15:00 | 546.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-25 11:15:00 | 467.25 | 2025-05-06 12:15:00 | 443.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 466.00 | 2025-05-06 12:15:00 | 444.31 | PARTIAL | 0.50 | 4.65% |
| SELL | retest2 | 2025-04-30 12:00:00 | 467.70 | 2025-05-06 13:15:00 | 442.70 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-04-30 13:00:00 | 459.00 | 2025-05-06 14:15:00 | 436.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 11:15:00 | 467.25 | 2025-05-07 10:15:00 | 444.75 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-04-30 09:15:00 | 466.00 | 2025-05-07 10:15:00 | 444.75 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2025-04-30 12:00:00 | 467.70 | 2025-05-07 10:15:00 | 444.75 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-04-30 13:00:00 | 459.00 | 2025-05-07 10:15:00 | 444.75 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2025-05-06 12:15:00 | 446.50 | 2025-05-09 09:15:00 | 424.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 13:00:00 | 445.00 | 2025-05-09 09:15:00 | 422.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 12:15:00 | 446.50 | 2025-05-09 15:15:00 | 430.50 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-05-06 13:00:00 | 445.00 | 2025-05-09 15:15:00 | 430.50 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2025-05-07 11:45:00 | 442.55 | 2025-05-12 10:15:00 | 459.40 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-05-21 12:00:00 | 456.05 | 2025-05-28 14:15:00 | 459.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-22 14:15:00 | 458.45 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-05-23 09:45:00 | 458.80 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-05-23 10:45:00 | 455.30 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-26 10:45:00 | 455.85 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-26 11:15:00 | 453.85 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-26 12:15:00 | 455.60 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-05-26 13:15:00 | 454.55 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-05-28 11:15:00 | 454.00 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-03 10:15:00 | 491.10 | 2025-06-04 10:15:00 | 540.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 09:15:00 | 591.05 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2025-06-18 09:45:00 | 562.05 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-18 10:30:00 | 559.95 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-18 11:00:00 | 560.50 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-19 11:30:00 | 554.10 | 2025-06-20 09:15:00 | 563.35 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-06-25 09:15:00 | 589.10 | 2025-07-01 14:15:00 | 592.50 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-07-10 10:45:00 | 590.00 | 2025-07-14 09:15:00 | 593.10 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-10 12:00:00 | 589.50 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-07-11 09:15:00 | 590.30 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-11 10:30:00 | 590.00 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-07-14 09:15:00 | 589.25 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-14 11:30:00 | 589.50 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-24 10:15:00 | 637.90 | 2025-07-25 12:15:00 | 616.35 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-07-30 10:15:00 | 604.25 | 2025-07-30 14:15:00 | 611.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-08-14 09:15:00 | 618.70 | 2025-08-14 12:15:00 | 582.20 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest2 | 2025-08-26 14:30:00 | 667.10 | 2025-08-28 14:15:00 | 657.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-08-28 11:00:00 | 673.85 | 2025-08-28 14:15:00 | 657.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-08-29 10:15:00 | 667.80 | 2025-09-04 12:15:00 | 693.70 | STOP_HIT | 1.00 | 3.88% |
| SELL | retest2 | 2025-09-08 15:00:00 | 685.95 | 2025-09-09 11:15:00 | 702.80 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-09-09 09:30:00 | 683.30 | 2025-09-09 11:15:00 | 702.80 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-09-09 11:15:00 | 685.70 | 2025-09-09 11:15:00 | 702.80 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-09-15 09:15:00 | 712.70 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-15 13:15:00 | 707.70 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-15 14:30:00 | 709.00 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-16 09:15:00 | 712.25 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-22 09:30:00 | 724.40 | 2025-09-22 14:15:00 | 709.35 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-26 11:15:00 | 689.30 | 2025-09-30 10:15:00 | 654.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 11:15:00 | 689.30 | 2025-10-01 15:15:00 | 647.90 | STOP_HIT | 0.50 | 6.01% |
| BUY | retest2 | 2025-10-29 13:30:00 | 625.15 | 2025-11-03 11:15:00 | 612.15 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-11-06 10:30:00 | 594.20 | 2025-11-10 09:15:00 | 564.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 11:00:00 | 595.45 | 2025-11-10 09:15:00 | 565.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 15:00:00 | 595.20 | 2025-11-10 09:15:00 | 565.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:30:00 | 594.20 | 2025-11-11 10:15:00 | 534.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 11:00:00 | 595.45 | 2025-11-11 10:15:00 | 535.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 15:00:00 | 595.20 | 2025-11-11 10:15:00 | 535.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 536.50 | 2025-11-24 14:15:00 | 509.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 536.50 | 2025-11-25 13:15:00 | 514.45 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-12-02 12:30:00 | 520.25 | 2025-12-03 09:15:00 | 525.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-03 10:45:00 | 521.00 | 2025-12-03 11:15:00 | 527.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-16 12:15:00 | 572.30 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-16 13:00:00 | 571.55 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-16 13:45:00 | 571.30 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-16 14:30:00 | 571.50 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-18 11:45:00 | 570.90 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-19 09:15:00 | 573.20 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-23 09:45:00 | 573.65 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-02 10:45:00 | 568.90 | 2026-01-06 13:15:00 | 625.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 09:30:00 | 568.05 | 2026-01-06 13:15:00 | 624.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 10:00:00 | 569.85 | 2026-01-06 13:15:00 | 626.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 10:30:00 | 571.95 | 2026-01-08 15:15:00 | 583.90 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2026-01-16 11:45:00 | 565.70 | 2026-01-20 14:15:00 | 537.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 565.55 | 2026-01-20 14:15:00 | 537.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:45:00 | 565.00 | 2026-01-20 14:15:00 | 536.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 565.80 | 2026-01-20 14:15:00 | 537.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 565.70 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2026-01-16 12:15:00 | 565.55 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-01-16 12:45:00 | 565.00 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2026-01-16 14:15:00 | 565.80 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2026-01-20 09:15:00 | 551.30 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-01-20 10:30:00 | 551.20 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-01-20 12:00:00 | 550.50 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-22 10:15:00 | 551.60 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-01-29 13:30:00 | 549.50 | 2026-02-01 12:15:00 | 542.65 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-01-29 14:15:00 | 549.50 | 2026-02-01 12:15:00 | 542.65 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-11 09:30:00 | 574.75 | 2026-02-11 10:15:00 | 567.20 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2026-03-05 11:00:00 | 563.50 | 2026-03-05 14:15:00 | 572.05 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest1 | 2026-03-05 11:45:00 | 562.75 | 2026-03-05 14:15:00 | 572.05 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-03-06 15:15:00 | 566.00 | 2026-03-09 09:15:00 | 537.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 566.00 | 2026-03-10 09:15:00 | 555.95 | STOP_HIT | 0.50 | 1.78% |
| BUY | retest2 | 2026-03-12 10:45:00 | 577.60 | 2026-03-13 10:15:00 | 556.30 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-03-12 15:00:00 | 576.55 | 2026-03-13 10:15:00 | 556.30 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-03-17 11:15:00 | 546.35 | 2026-03-18 09:15:00 | 561.50 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-04-06 11:45:00 | 563.85 | 2026-04-15 09:15:00 | 620.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:15:00 | 562.10 | 2026-04-15 09:15:00 | 618.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 14:45:00 | 564.00 | 2026-04-15 09:15:00 | 620.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 09:45:00 | 564.80 | 2026-04-15 09:15:00 | 621.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 573.35 | 2026-04-17 09:15:00 | 630.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 14:00:00 | 667.55 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-29 11:00:00 | 668.00 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-29 13:15:00 | 668.35 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-04-29 14:45:00 | 667.50 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.21% |
