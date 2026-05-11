# Life Insurance Corporation of India (LICI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 802.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 225 |
| ALERT1 | 149 |
| ALERT2 | 148 |
| ALERT2_SKIP | 99 |
| ALERT3 | 321 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 140 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 141 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 152 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 111
- **Target hits / Stop hits / Partials:** 0 / 141 / 11
- **Avg / median % per leg:** -0.13% / -0.82%
- **Sum % (uncompounded):** -19.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 8 | 16.3% | 0 | 49 | 0 | -0.70% | -34.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.04% | -2.0% |
| BUY @ 3rd Alert (retest2) | 48 | 8 | 16.7% | 0 | 48 | 0 | -0.68% | -32.5% |
| SELL (all) | 103 | 33 | 32.0% | 0 | 92 | 11 | 0.15% | 15.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 103 | 33 | 32.0% | 0 | 92 | 11 | 0.15% | 15.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.04% | -2.0% |
| retest2 (combined) | 151 | 41 | 27.2% | 0 | 140 | 11 | -0.11% | -17.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 563.90 | 568.33 | 568.42 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 574.50 | 568.61 | 567.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 576.85 | 572.52 | 570.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 15:15:00 | 593.70 | 594.05 | 588.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 604.00 | 604.75 | 603.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 604.00 | 604.75 | 603.08 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 596.70 | 601.43 | 602.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 11:15:00 | 595.00 | 599.16 | 600.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 598.00 | 597.16 | 599.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 11:15:00 | 597.15 | 597.01 | 598.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 11:15:00 | 597.15 | 597.01 | 598.61 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 10:15:00 | 600.30 | 598.83 | 598.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 14:15:00 | 596.10 | 598.24 | 598.48 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 600.50 | 598.18 | 598.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 603.10 | 599.16 | 598.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 15:15:00 | 600.85 | 600.91 | 599.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 14:15:00 | 602.15 | 603.98 | 603.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 602.15 | 603.98 | 603.16 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 596.90 | 602.28 | 602.51 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 11:15:00 | 604.50 | 599.38 | 598.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 13:15:00 | 605.00 | 601.17 | 599.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 631.45 | 638.26 | 633.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 631.45 | 638.26 | 633.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 631.45 | 638.26 | 633.71 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 14:15:00 | 625.65 | 630.86 | 631.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 15:15:00 | 623.85 | 629.46 | 630.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 14:15:00 | 617.55 | 616.44 | 620.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 621.35 | 617.47 | 620.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 621.35 | 617.47 | 620.31 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 625.50 | 621.09 | 620.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 13:15:00 | 630.45 | 624.14 | 622.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 14:15:00 | 630.00 | 630.09 | 627.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 626.70 | 629.21 | 627.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 626.70 | 629.21 | 627.32 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 11:15:00 | 625.35 | 626.77 | 626.87 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 629.25 | 627.11 | 626.96 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 624.80 | 626.97 | 627.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 11:15:00 | 623.15 | 626.21 | 626.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 624.90 | 621.17 | 622.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 624.90 | 621.17 | 622.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 624.90 | 621.17 | 622.69 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 622.95 | 619.70 | 619.69 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 619.35 | 620.14 | 620.19 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 14:15:00 | 620.60 | 620.23 | 620.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 15:15:00 | 622.10 | 620.60 | 620.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 10:15:00 | 620.55 | 620.70 | 620.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 10:15:00 | 620.55 | 620.70 | 620.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 620.55 | 620.70 | 620.49 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 15:15:00 | 620.05 | 620.33 | 620.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 09:15:00 | 619.20 | 620.10 | 620.26 | Break + close below crossover candle low |

### Cycle 18 — BUY (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 10:15:00 | 628.05 | 621.69 | 620.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 11:15:00 | 636.80 | 628.39 | 625.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 10:15:00 | 634.45 | 634.55 | 630.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 13:15:00 | 631.75 | 633.58 | 630.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 631.75 | 633.58 | 630.96 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 623.35 | 628.87 | 629.29 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 630.05 | 628.05 | 627.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 13:15:00 | 635.10 | 630.28 | 628.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 15:15:00 | 630.00 | 630.34 | 629.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 09:15:00 | 627.30 | 629.73 | 629.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 627.30 | 629.73 | 629.07 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 12:15:00 | 627.00 | 628.49 | 628.60 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 635.55 | 629.85 | 629.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 639.75 | 633.93 | 631.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 645.80 | 646.27 | 641.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 12:15:00 | 642.15 | 645.21 | 641.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 642.15 | 645.21 | 641.54 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 12:15:00 | 643.70 | 650.41 | 651.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 15:15:00 | 640.70 | 643.74 | 645.51 | Break + close below crossover candle low |

### Cycle 24 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 661.60 | 647.31 | 646.97 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 12:15:00 | 650.60 | 652.98 | 653.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 14:15:00 | 649.20 | 651.81 | 652.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 654.30 | 651.85 | 652.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 654.30 | 651.85 | 652.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 654.30 | 651.85 | 652.43 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 657.45 | 652.97 | 652.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 11:15:00 | 660.00 | 654.38 | 653.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 09:15:00 | 654.30 | 656.78 | 655.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 654.30 | 656.78 | 655.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 654.30 | 656.78 | 655.32 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 13:15:00 | 652.95 | 654.50 | 654.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 15:15:00 | 650.05 | 653.16 | 653.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 15:15:00 | 653.25 | 652.52 | 653.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 15:15:00 | 653.25 | 652.52 | 653.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 653.25 | 652.52 | 653.12 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 658.90 | 653.91 | 653.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 665.70 | 656.26 | 654.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 659.80 | 660.58 | 658.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 14:15:00 | 658.55 | 660.47 | 658.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 658.55 | 660.47 | 658.85 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 655.00 | 658.48 | 658.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 648.50 | 655.80 | 657.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 11:15:00 | 654.95 | 651.41 | 653.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 11:15:00 | 654.95 | 651.41 | 653.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 654.95 | 651.41 | 653.50 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 658.30 | 654.58 | 654.43 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 652.90 | 654.15 | 654.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 15:15:00 | 652.00 | 653.44 | 653.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 654.00 | 653.55 | 653.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 654.00 | 653.55 | 653.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 654.00 | 653.55 | 653.92 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 654.75 | 654.13 | 654.12 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 13:15:00 | 653.40 | 653.98 | 654.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 14:15:00 | 651.50 | 653.49 | 653.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 653.90 | 653.14 | 653.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 653.90 | 653.14 | 653.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 653.90 | 653.14 | 653.58 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 10:15:00 | 658.35 | 650.75 | 650.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 666.30 | 658.62 | 655.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 14:15:00 | 660.05 | 661.47 | 658.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 12:15:00 | 675.55 | 676.98 | 674.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 675.55 | 676.98 | 674.64 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 666.60 | 674.89 | 675.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 661.80 | 670.50 | 673.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 661.95 | 660.77 | 664.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 661.95 | 660.77 | 664.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 661.95 | 660.77 | 664.54 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 15:15:00 | 668.40 | 664.40 | 663.91 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 660.20 | 664.36 | 664.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 11:15:00 | 658.95 | 662.65 | 663.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 658.40 | 658.25 | 660.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 652.00 | 655.54 | 658.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 652.00 | 655.54 | 658.15 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 648.95 | 647.74 | 647.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 14:15:00 | 649.85 | 648.61 | 648.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 645.60 | 648.30 | 648.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 645.60 | 648.30 | 648.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 645.60 | 648.30 | 648.06 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 10:15:00 | 645.40 | 647.72 | 647.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 640.30 | 645.08 | 646.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 639.75 | 639.57 | 642.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 645.05 | 640.34 | 641.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 645.05 | 640.34 | 641.34 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 11:15:00 | 644.95 | 641.98 | 641.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 13:15:00 | 645.50 | 643.06 | 642.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 15:15:00 | 643.35 | 643.36 | 642.73 | EMA200 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 634.85 | 641.66 | 642.01 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 11:15:00 | 641.45 | 638.89 | 638.74 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 14:15:00 | 637.35 | 638.70 | 638.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 636.95 | 638.35 | 638.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 637.85 | 636.47 | 637.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 637.85 | 636.47 | 637.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 637.85 | 636.47 | 637.26 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 12:15:00 | 639.00 | 637.93 | 637.82 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 14:15:00 | 636.05 | 637.42 | 637.60 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 15:15:00 | 639.10 | 637.76 | 637.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 651.10 | 640.43 | 638.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 642.60 | 645.44 | 643.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 642.60 | 645.44 | 643.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 642.60 | 645.44 | 643.12 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 14:15:00 | 637.95 | 642.10 | 642.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 15:15:00 | 635.45 | 640.77 | 641.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 15:15:00 | 635.45 | 635.39 | 636.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 09:15:00 | 628.80 | 634.07 | 636.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 628.80 | 634.07 | 636.11 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 607.35 | 604.72 | 604.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 609.55 | 606.15 | 605.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 610.90 | 611.24 | 609.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 12:15:00 | 614.20 | 615.24 | 614.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 614.20 | 615.24 | 614.27 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 15:15:00 | 609.80 | 612.99 | 613.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-12 18:15:00 | 608.00 | 611.99 | 612.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 12:15:00 | 607.75 | 606.93 | 608.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 605.85 | 606.76 | 607.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 605.85 | 606.76 | 607.91 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-11-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 13:15:00 | 610.05 | 608.77 | 608.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 09:15:00 | 617.15 | 610.58 | 609.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 10:15:00 | 612.55 | 613.41 | 611.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 11:15:00 | 610.90 | 612.91 | 611.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 610.90 | 612.91 | 611.89 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 611.30 | 611.91 | 611.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 10:15:00 | 610.35 | 611.60 | 611.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 610.00 | 609.87 | 610.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 610.00 | 609.87 | 610.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 610.00 | 609.87 | 610.71 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 15:15:00 | 623.10 | 612.78 | 611.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 09:15:00 | 640.85 | 618.40 | 614.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 15:15:00 | 672.15 | 674.29 | 659.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 12:15:00 | 677.60 | 681.06 | 677.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 677.60 | 681.06 | 677.77 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 09:15:00 | 793.35 | 800.43 | 801.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 791.10 | 797.08 | 798.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 766.80 | 766.57 | 776.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 804.50 | 774.15 | 779.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 804.50 | 774.15 | 779.12 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 803.20 | 784.55 | 783.27 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 15:15:00 | 782.05 | 785.84 | 786.08 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 792.85 | 787.24 | 786.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 11:15:00 | 811.90 | 793.34 | 789.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 846.70 | 849.79 | 838.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 13:15:00 | 841.10 | 845.46 | 839.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 841.10 | 845.46 | 839.63 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 14:15:00 | 836.60 | 839.15 | 839.46 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 854.00 | 841.47 | 840.42 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 829.60 | 840.82 | 841.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 823.70 | 833.39 | 837.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 834.60 | 832.13 | 836.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 834.60 | 832.13 | 836.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 834.60 | 832.13 | 836.07 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 10:15:00 | 840.85 | 836.48 | 836.09 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 13:15:00 | 832.85 | 836.69 | 836.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 10:15:00 | 829.70 | 833.75 | 835.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 848.85 | 834.33 | 834.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 848.85 | 834.33 | 834.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 848.85 | 834.33 | 834.46 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 10:15:00 | 843.55 | 836.18 | 835.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 12:15:00 | 855.05 | 842.16 | 838.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 14:15:00 | 889.05 | 892.43 | 878.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 882.00 | 889.64 | 879.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 882.00 | 889.64 | 879.89 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 874.40 | 905.54 | 907.44 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 913.50 | 901.78 | 901.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 945.45 | 920.51 | 912.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 933.60 | 934.61 | 924.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 10:15:00 | 943.75 | 948.68 | 940.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 943.75 | 948.68 | 940.62 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 1042.95 | 1068.00 | 1068.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 1019.00 | 1055.97 | 1062.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 1019.00 | 1017.95 | 1033.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 1024.75 | 1019.31 | 1032.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 1024.75 | 1019.31 | 1032.41 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 1064.00 | 1039.27 | 1038.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 15:15:00 | 1077.00 | 1046.82 | 1041.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 14:15:00 | 1055.30 | 1062.01 | 1053.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 14:15:00 | 1055.30 | 1062.01 | 1053.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 1055.30 | 1062.01 | 1053.60 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 12:15:00 | 1039.95 | 1048.45 | 1049.54 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 1058.00 | 1049.14 | 1048.91 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 1039.80 | 1049.34 | 1049.95 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 11:15:00 | 1056.85 | 1046.94 | 1046.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 12:15:00 | 1059.00 | 1049.35 | 1047.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 1058.00 | 1064.07 | 1059.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 1058.00 | 1064.07 | 1059.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1058.00 | 1064.07 | 1059.02 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 1046.80 | 1054.93 | 1055.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 1040.40 | 1052.03 | 1054.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 1050.60 | 1050.06 | 1052.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 1050.60 | 1050.06 | 1052.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1050.60 | 1050.06 | 1052.98 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 1033.60 | 1029.21 | 1029.00 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 1022.90 | 1029.33 | 1029.67 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 12:15:00 | 1039.50 | 1031.36 | 1030.56 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 15:15:00 | 1028.00 | 1032.53 | 1032.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1011.30 | 1028.28 | 1030.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 1015.30 | 1013.84 | 1020.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 10:15:00 | 1020.00 | 1015.08 | 1020.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 1020.00 | 1015.08 | 1020.21 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 1037.45 | 1023.82 | 1022.60 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 13:15:00 | 1016.05 | 1020.89 | 1021.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 1012.90 | 1019.29 | 1020.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 936.65 | 935.44 | 960.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 956.30 | 941.36 | 955.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 956.30 | 941.36 | 955.27 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 902.00 | 893.45 | 893.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 905.60 | 897.83 | 895.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 900.85 | 906.41 | 902.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 14:15:00 | 900.85 | 906.41 | 902.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 900.85 | 906.41 | 902.76 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 15:15:00 | 893.05 | 900.95 | 901.48 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 924.30 | 905.62 | 903.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 09:15:00 | 931.90 | 917.99 | 911.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 13:15:00 | 1001.70 | 1001.77 | 989.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 13:15:00 | 995.70 | 999.07 | 994.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 995.70 | 999.07 | 994.36 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 980.00 | 994.12 | 995.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 15:15:00 | 976.10 | 988.53 | 992.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 969.00 | 966.90 | 972.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 969.00 | 966.90 | 972.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 969.00 | 966.90 | 972.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 973.00 | 966.90 | 972.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 981.50 | 969.82 | 972.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 981.50 | 969.82 | 972.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 977.00 | 971.26 | 973.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 981.80 | 971.26 | 973.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 975.50 | 972.34 | 973.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:45:00 | 974.65 | 972.34 | 973.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 976.25 | 973.13 | 973.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:00:00 | 976.25 | 973.13 | 973.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 976.05 | 973.71 | 973.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 982.75 | 973.71 | 973.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 976.80 | 974.18 | 974.11 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 970.50 | 973.87 | 974.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 961.05 | 971.31 | 972.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 970.00 | 968.26 | 970.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 12:15:00 | 970.00 | 968.26 | 970.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 970.00 | 968.26 | 970.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:45:00 | 970.00 | 968.26 | 970.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 974.70 | 969.55 | 970.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 13:45:00 | 974.25 | 969.55 | 970.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 974.25 | 970.49 | 971.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 973.40 | 970.49 | 971.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 972.60 | 970.91 | 971.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:15:00 | 980.00 | 970.91 | 971.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 980.00 | 972.73 | 972.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 1000.90 | 980.41 | 976.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 12:15:00 | 981.55 | 981.75 | 978.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 13:00:00 | 981.55 | 981.75 | 978.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 986.05 | 990.42 | 986.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 986.30 | 990.42 | 986.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 982.00 | 988.74 | 986.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 980.65 | 988.74 | 986.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 980.25 | 987.04 | 985.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:45:00 | 984.25 | 985.79 | 985.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:00:00 | 984.35 | 987.80 | 987.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 10:15:00 | 981.75 | 986.59 | 986.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 10:15:00 | 981.75 | 986.59 | 986.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 13:15:00 | 979.90 | 983.65 | 985.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 983.10 | 982.67 | 984.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 983.10 | 982.67 | 984.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 983.10 | 982.67 | 984.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 986.85 | 982.67 | 984.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 979.75 | 982.09 | 983.84 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 09:15:00 | 1002.40 | 984.65 | 983.92 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 978.85 | 987.25 | 988.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 951.00 | 977.79 | 983.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 929.55 | 928.62 | 943.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 929.55 | 928.62 | 943.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 913.90 | 913.96 | 923.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:30:00 | 907.00 | 910.91 | 920.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 14:00:00 | 907.25 | 910.18 | 918.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 905.45 | 910.44 | 917.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:30:00 | 907.60 | 897.68 | 905.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 911.10 | 900.36 | 905.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:00:00 | 911.10 | 900.36 | 905.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 917.90 | 903.87 | 906.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:00:00 | 917.90 | 903.87 | 906.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-14 13:15:00 | 924.15 | 910.87 | 909.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 924.15 | 910.87 | 909.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 926.85 | 914.06 | 911.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 970.25 | 971.17 | 955.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:30:00 | 969.25 | 971.17 | 955.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1019.25 | 1031.46 | 1024.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 1017.55 | 1031.46 | 1024.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1029.75 | 1031.12 | 1025.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:00:00 | 1044.35 | 1033.77 | 1026.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:30:00 | 1051.90 | 1034.39 | 1030.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 13:00:00 | 1039.10 | 1036.23 | 1031.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1043.00 | 1037.69 | 1033.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1018.35 | 1033.82 | 1032.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 1018.35 | 1033.82 | 1032.23 | SL hit (close<static) qty=1.00 sl=1019.05 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1023.50 | 1031.11 | 1031.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 1017.30 | 1027.22 | 1029.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 1006.50 | 999.82 | 1006.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 1006.50 | 999.82 | 1006.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1006.50 | 999.82 | 1006.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 1009.50 | 999.82 | 1006.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1006.15 | 1001.08 | 1006.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:15:00 | 1008.75 | 1001.08 | 1006.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 1009.25 | 1002.72 | 1006.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 1008.20 | 1002.72 | 1006.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1014.95 | 1005.16 | 1007.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 1014.95 | 1005.16 | 1007.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1012.00 | 1006.87 | 1007.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1038.40 | 1006.87 | 1007.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1051.35 | 1015.77 | 1011.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 1066.95 | 1040.39 | 1026.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1008.85 | 1038.16 | 1028.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1008.85 | 1038.16 | 1028.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1008.85 | 1038.16 | 1028.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 997.20 | 1038.16 | 1028.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 964.00 | 1023.33 | 1022.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 964.00 | 1023.33 | 1022.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 921.00 | 1002.87 | 1013.03 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 984.55 | 972.45 | 971.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 992.35 | 976.43 | 973.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 996.30 | 997.10 | 989.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 998.45 | 997.10 | 989.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1001.05 | 1005.31 | 1001.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 1001.05 | 1005.31 | 1001.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 999.00 | 1004.05 | 1000.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 999.00 | 1004.05 | 1000.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 999.00 | 1003.04 | 1000.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1008.65 | 1003.04 | 1000.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:30:00 | 999.95 | 1002.78 | 1001.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:30:00 | 1000.40 | 1001.65 | 1000.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 1000.30 | 1001.02 | 1000.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1034.00 | 1050.04 | 1040.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1037.30 | 1050.04 | 1040.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1032.10 | 1046.45 | 1039.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 1025.90 | 1046.45 | 1039.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 1033.10 | 1040.28 | 1038.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 1032.00 | 1040.28 | 1038.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1032.75 | 1037.00 | 1036.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 1034.75 | 1037.00 | 1036.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 1032.00 | 1036.00 | 1036.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 1032.00 | 1036.00 | 1036.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 1018.10 | 1028.52 | 1031.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 1012.85 | 1012.07 | 1017.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 1012.85 | 1012.07 | 1017.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1012.85 | 1012.07 | 1017.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 1017.85 | 1012.07 | 1017.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1012.50 | 1012.16 | 1016.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 1014.20 | 1012.16 | 1016.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1007.40 | 1006.74 | 1011.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 1009.75 | 1006.74 | 1011.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 998.50 | 998.83 | 1004.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 996.70 | 998.83 | 1004.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1003.00 | 999.66 | 1004.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 995.75 | 999.44 | 1003.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 13:30:00 | 994.00 | 998.26 | 1002.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 994.90 | 996.36 | 1001.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 997.55 | 998.40 | 1000.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 996.95 | 998.11 | 999.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:30:00 | 998.50 | 998.11 | 999.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 990.45 | 996.08 | 998.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 988.75 | 993.61 | 997.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:00:00 | 987.80 | 993.61 | 997.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 15:00:00 | 985.05 | 989.50 | 994.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 09:45:00 | 988.70 | 989.56 | 993.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 989.80 | 989.61 | 992.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 988.60 | 989.61 | 992.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 989.00 | 989.65 | 992.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 13:30:00 | 988.95 | 989.47 | 992.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 14:00:00 | 988.75 | 989.47 | 992.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 995.00 | 990.40 | 991.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 995.00 | 990.40 | 991.83 | SL hit (close>static) qty=1.00 sl=993.05 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 1018.70 | 996.06 | 994.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 12:15:00 | 1034.00 | 1016.07 | 1011.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 1050.05 | 1053.08 | 1046.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 15:00:00 | 1050.05 | 1053.08 | 1046.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 1050.00 | 1052.46 | 1047.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 1046.50 | 1052.46 | 1047.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1055.55 | 1053.08 | 1047.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 1040.50 | 1053.08 | 1047.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1066.90 | 1095.52 | 1088.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 1066.90 | 1095.52 | 1088.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1081.00 | 1092.62 | 1087.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 13:45:00 | 1087.50 | 1089.38 | 1087.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 1095.00 | 1118.89 | 1113.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 1085.00 | 1108.93 | 1109.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 1085.00 | 1108.93 | 1109.34 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1122.80 | 1109.63 | 1109.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1149.00 | 1121.87 | 1115.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 1174.75 | 1177.06 | 1164.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 12:00:00 | 1174.75 | 1177.06 | 1164.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 1171.90 | 1176.03 | 1165.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 13:30:00 | 1179.30 | 1174.54 | 1165.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 1180.10 | 1173.83 | 1166.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:30:00 | 1180.95 | 1174.24 | 1169.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:00:00 | 1181.50 | 1178.70 | 1173.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 1181.10 | 1179.18 | 1174.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 1174.10 | 1179.18 | 1174.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1172.40 | 1177.18 | 1174.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 1172.40 | 1177.18 | 1174.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1172.00 | 1176.14 | 1173.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1172.00 | 1176.14 | 1173.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1177.00 | 1176.32 | 1174.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1183.60 | 1176.32 | 1174.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:45:00 | 1185.00 | 1184.75 | 1183.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 12:15:00 | 1174.85 | 1180.90 | 1181.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 1174.85 | 1180.90 | 1181.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 1170.60 | 1178.84 | 1180.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 15:15:00 | 1182.70 | 1178.35 | 1179.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 15:15:00 | 1182.70 | 1178.35 | 1179.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 1182.70 | 1178.35 | 1179.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1129.25 | 1178.35 | 1179.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 1134.90 | 1124.63 | 1123.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 1134.90 | 1124.63 | 1123.58 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 1085.35 | 1124.12 | 1126.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 1074.00 | 1100.43 | 1112.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 14:15:00 | 1030.10 | 1029.45 | 1053.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 15:00:00 | 1030.10 | 1029.45 | 1053.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1040.45 | 1032.51 | 1050.42 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1067.10 | 1054.24 | 1053.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 1077.25 | 1067.68 | 1061.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 1069.00 | 1070.57 | 1065.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 1069.00 | 1070.57 | 1065.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1069.00 | 1070.57 | 1065.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 1069.00 | 1070.57 | 1065.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1067.95 | 1070.05 | 1065.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 1070.05 | 1070.05 | 1065.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1076.25 | 1071.29 | 1066.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 12:15:00 | 1079.45 | 1073.32 | 1068.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:15:00 | 1079.60 | 1074.65 | 1070.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:45:00 | 1081.10 | 1076.52 | 1071.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:45:00 | 1080.10 | 1079.02 | 1074.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1075.60 | 1077.53 | 1074.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 1072.95 | 1077.53 | 1074.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1076.90 | 1077.40 | 1074.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1073.80 | 1077.40 | 1074.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1068.15 | 1075.55 | 1074.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 1068.15 | 1075.55 | 1074.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1069.05 | 1074.25 | 1073.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1070.15 | 1074.25 | 1073.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 1067.50 | 1072.90 | 1073.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 1067.50 | 1072.90 | 1073.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 1062.80 | 1070.15 | 1071.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 1075.10 | 1059.85 | 1063.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 10:15:00 | 1075.10 | 1059.85 | 1063.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1075.10 | 1059.85 | 1063.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 1075.10 | 1059.85 | 1063.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 1072.95 | 1062.47 | 1063.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:15:00 | 1077.60 | 1062.47 | 1063.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 1083.45 | 1066.67 | 1065.70 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1067.90 | 1074.62 | 1074.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1063.50 | 1072.39 | 1073.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1063.60 | 1062.13 | 1065.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 1063.60 | 1062.13 | 1065.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1063.60 | 1062.13 | 1065.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:00:00 | 1059.35 | 1061.57 | 1064.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:30:00 | 1060.10 | 1060.18 | 1063.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 1060.00 | 1061.18 | 1063.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 1065.00 | 1062.21 | 1062.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 1065.00 | 1062.21 | 1062.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 1067.85 | 1063.34 | 1062.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1056.30 | 1064.60 | 1063.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1056.30 | 1064.60 | 1063.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1056.30 | 1064.60 | 1063.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1056.30 | 1064.60 | 1063.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1045.80 | 1060.84 | 1062.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1040.00 | 1050.80 | 1056.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 1035.00 | 1032.12 | 1041.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 1040.40 | 1032.12 | 1041.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1034.30 | 1032.56 | 1040.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1031.60 | 1032.55 | 1037.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 1041.05 | 1030.67 | 1030.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1041.05 | 1030.67 | 1030.55 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1027.95 | 1031.77 | 1032.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1022.15 | 1028.76 | 1030.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 1028.05 | 1024.51 | 1026.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 1028.05 | 1024.51 | 1026.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1028.05 | 1024.51 | 1026.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 1030.75 | 1024.51 | 1026.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1031.00 | 1025.81 | 1027.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 1031.05 | 1025.81 | 1027.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1024.60 | 1025.56 | 1027.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:45:00 | 1023.30 | 1024.73 | 1026.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:30:00 | 1022.55 | 1020.29 | 1023.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 1026.30 | 1016.39 | 1015.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1026.30 | 1016.39 | 1015.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 1034.00 | 1024.84 | 1020.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 1026.35 | 1029.88 | 1025.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 1026.35 | 1029.88 | 1025.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1027.45 | 1029.39 | 1025.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 1029.60 | 1029.39 | 1025.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1020.65 | 1027.64 | 1025.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1020.65 | 1027.64 | 1025.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1019.65 | 1026.04 | 1024.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 1018.80 | 1026.04 | 1024.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 1017.25 | 1022.97 | 1023.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 13:15:00 | 1015.25 | 1021.42 | 1022.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 15:15:00 | 1021.90 | 1020.93 | 1022.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:15:00 | 1015.25 | 1020.93 | 1022.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 110 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 1035.80 | 1023.90 | 1023.57 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 13:15:00 | 1021.00 | 1024.74 | 1024.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1010.05 | 1021.89 | 1023.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 980.90 | 978.37 | 988.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 980.90 | 978.37 | 988.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 959.40 | 946.36 | 954.78 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 971.40 | 959.59 | 959.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 974.25 | 968.58 | 964.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 969.60 | 969.80 | 966.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 969.60 | 969.80 | 966.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 965.65 | 968.85 | 966.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 965.65 | 968.85 | 966.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 965.25 | 968.13 | 966.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:45:00 | 968.40 | 968.62 | 966.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 957.05 | 965.84 | 965.71 | SL hit (close<static) qty=1.00 sl=962.05 alert=retest2 |

### Cycle 113 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 952.80 | 963.23 | 964.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 949.05 | 960.39 | 963.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 963.00 | 957.99 | 961.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 963.00 | 957.99 | 961.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 963.00 | 957.99 | 961.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 963.00 | 957.99 | 961.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 959.35 | 958.26 | 960.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:00:00 | 957.25 | 958.50 | 960.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 956.80 | 958.88 | 960.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 909.39 | 920.60 | 927.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 908.96 | 917.15 | 923.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 918.05 | 917.33 | 923.25 | SL hit (close>ema200) qty=0.50 sl=917.33 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 920.55 | 910.41 | 910.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 925.10 | 914.76 | 912.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 934.25 | 934.92 | 927.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 934.25 | 934.92 | 927.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 929.00 | 933.28 | 929.05 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 923.40 | 927.26 | 927.34 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 929.05 | 927.62 | 927.50 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 914.40 | 925.47 | 926.57 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 929.45 | 924.50 | 924.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 931.40 | 926.57 | 925.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 937.95 | 939.06 | 934.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 937.95 | 939.06 | 934.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 929.80 | 936.70 | 934.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 929.80 | 936.70 | 934.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 928.00 | 934.96 | 933.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 928.55 | 934.96 | 933.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 919.50 | 930.64 | 931.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 918.00 | 928.11 | 930.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 925.85 | 922.02 | 925.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 925.85 | 922.02 | 925.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 925.85 | 922.02 | 925.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:15:00 | 927.55 | 922.02 | 925.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 927.95 | 923.20 | 925.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 928.15 | 923.20 | 925.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 924.15 | 923.39 | 925.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 921.30 | 923.39 | 925.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 953.50 | 927.61 | 926.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 953.50 | 927.61 | 926.68 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 921.25 | 927.43 | 927.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 898.35 | 921.61 | 925.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 903.60 | 903.34 | 911.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 12:15:00 | 910.35 | 904.18 | 909.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 910.35 | 904.18 | 909.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 910.35 | 904.18 | 909.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 919.70 | 907.28 | 910.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:00:00 | 919.70 | 907.28 | 910.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 905.55 | 906.94 | 910.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 915.30 | 906.94 | 910.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 909.00 | 907.45 | 910.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 899.00 | 905.21 | 907.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 904.45 | 896.83 | 895.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 904.45 | 896.83 | 895.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 912.60 | 903.65 | 899.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 13:15:00 | 906.90 | 907.05 | 903.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 13:15:00 | 906.90 | 907.05 | 903.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 906.90 | 907.05 | 903.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 906.90 | 907.05 | 903.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 972.60 | 979.05 | 971.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:00:00 | 972.60 | 979.05 | 971.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 969.45 | 977.13 | 971.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 969.45 | 977.13 | 971.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 968.00 | 975.31 | 971.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 975.60 | 975.31 | 971.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:45:00 | 971.35 | 972.60 | 972.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 957.10 | 982.16 | 982.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 957.10 | 982.16 | 982.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 948.60 | 967.48 | 974.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 929.95 | 928.63 | 936.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 929.95 | 928.63 | 936.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 927.25 | 929.33 | 935.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 933.60 | 929.33 | 935.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 930.15 | 926.56 | 930.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:30:00 | 926.45 | 926.86 | 930.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 880.13 | 887.39 | 890.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 905.25 | 890.97 | 892.21 | SL hit (close>ema200) qty=0.50 sl=890.97 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 912.50 | 895.27 | 894.05 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 885.45 | 891.86 | 892.63 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 896.15 | 892.21 | 892.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 897.00 | 893.17 | 892.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 894.50 | 904.18 | 901.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 894.50 | 904.18 | 901.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 894.50 | 904.18 | 901.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 894.40 | 904.18 | 901.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 883.00 | 899.94 | 900.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 880.05 | 891.33 | 895.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 819.35 | 819.16 | 829.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:00:00 | 819.35 | 819.16 | 829.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 823.25 | 822.07 | 828.73 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 849.95 | 833.83 | 831.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 855.90 | 845.13 | 839.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 15:15:00 | 857.05 | 858.27 | 852.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 860.05 | 857.29 | 853.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 842.50 | 854.33 | 852.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 842.50 | 854.33 | 852.07 | SL hit (close<ema400) qty=1.00 sl=852.07 alert=retest1 |

### Cycle 129 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 841.60 | 849.19 | 850.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 837.45 | 846.84 | 848.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 838.25 | 832.36 | 837.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 838.25 | 832.36 | 837.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 838.25 | 832.36 | 837.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 838.25 | 832.36 | 837.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 837.20 | 833.33 | 837.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 839.25 | 833.33 | 837.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 838.40 | 834.34 | 837.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 837.00 | 834.34 | 837.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 838.30 | 835.13 | 837.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:15:00 | 837.80 | 835.13 | 837.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 837.05 | 835.52 | 837.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 838.10 | 835.52 | 837.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 839.45 | 836.30 | 837.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 839.45 | 836.30 | 837.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 836.50 | 836.34 | 837.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 835.10 | 836.34 | 837.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 829.35 | 834.94 | 837.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 827.70 | 832.31 | 835.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 831.30 | 822.84 | 822.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 831.30 | 822.84 | 822.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 841.45 | 829.88 | 827.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 837.15 | 846.35 | 840.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 837.15 | 846.35 | 840.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 837.15 | 846.35 | 840.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 837.30 | 846.35 | 840.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 855.40 | 848.16 | 841.43 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 824.50 | 838.36 | 839.08 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 843.30 | 837.15 | 836.41 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 832.45 | 837.13 | 837.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 829.60 | 835.62 | 836.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 800.40 | 799.74 | 808.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 800.40 | 799.74 | 808.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 766.55 | 761.08 | 767.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 761.55 | 761.08 | 767.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 764.25 | 761.99 | 767.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 763.75 | 762.34 | 767.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 760.00 | 761.85 | 766.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 770.50 | 762.05 | 764.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 770.50 | 762.05 | 764.64 | SL hit (close>static) qty=1.00 sl=769.65 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 769.75 | 765.72 | 765.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 772.35 | 767.05 | 766.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 778.70 | 778.84 | 774.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 778.70 | 778.84 | 774.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 778.70 | 778.84 | 774.95 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 761.50 | 773.33 | 773.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 756.60 | 763.98 | 768.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 732.05 | 731.86 | 740.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 732.05 | 731.86 | 740.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 735.35 | 731.71 | 737.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 736.85 | 731.71 | 737.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 736.50 | 732.67 | 737.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 734.10 | 732.54 | 737.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 734.15 | 733.24 | 736.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 733.30 | 733.24 | 736.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 748.80 | 736.39 | 737.19 | SL hit (close>static) qty=1.00 sl=738.50 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 755.30 | 740.17 | 738.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 758.00 | 743.74 | 740.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 763.05 | 765.14 | 758.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 763.05 | 765.14 | 758.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 759.50 | 764.01 | 758.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 759.50 | 764.01 | 758.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 760.45 | 763.30 | 759.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:00:00 | 760.45 | 763.30 | 759.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 764.45 | 763.76 | 760.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 763.75 | 763.76 | 760.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 758.50 | 762.61 | 760.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 758.50 | 762.61 | 760.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 758.05 | 761.70 | 760.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:15:00 | 757.35 | 761.70 | 760.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 753.00 | 758.71 | 759.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 749.00 | 756.77 | 758.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 757.15 | 756.84 | 758.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 757.15 | 756.84 | 758.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 757.15 | 756.84 | 758.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 757.15 | 756.84 | 758.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 754.50 | 756.38 | 757.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 754.50 | 756.38 | 757.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 754.15 | 754.48 | 756.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 754.90 | 754.48 | 756.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 749.05 | 753.54 | 755.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 744.40 | 750.91 | 754.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 746.80 | 746.15 | 749.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 745.60 | 746.15 | 749.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:45:00 | 745.90 | 743.50 | 746.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 745.60 | 743.68 | 745.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 744.95 | 743.68 | 745.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 742.40 | 743.42 | 745.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 751.70 | 746.73 | 746.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 751.70 | 746.73 | 746.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 754.20 | 748.23 | 747.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 797.30 | 798.63 | 791.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 11:15:00 | 794.45 | 797.01 | 791.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 794.45 | 797.01 | 791.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 790.10 | 797.01 | 791.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 792.95 | 796.20 | 791.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:15:00 | 791.00 | 796.20 | 791.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 795.20 | 796.00 | 792.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 797.40 | 793.29 | 791.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 13:15:00 | 789.30 | 792.58 | 791.88 | SL hit (close<static) qty=1.00 sl=790.50 alert=retest2 |

### Cycle 139 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 785.80 | 790.39 | 790.95 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 802.85 | 792.53 | 791.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 811.00 | 797.67 | 794.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 801.00 | 801.15 | 796.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 801.00 | 801.15 | 796.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 798.40 | 800.53 | 797.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 798.80 | 800.53 | 797.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 802.30 | 800.88 | 798.13 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 788.80 | 797.06 | 797.45 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 801.70 | 797.99 | 797.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 802.55 | 798.90 | 798.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 796.45 | 808.84 | 806.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 796.45 | 808.84 | 806.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 796.45 | 808.84 | 806.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 796.45 | 808.84 | 806.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 801.25 | 807.32 | 805.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 801.00 | 807.32 | 805.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 796.40 | 803.65 | 804.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 790.50 | 801.02 | 803.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 775.60 | 772.66 | 782.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 775.60 | 772.66 | 782.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 775.60 | 772.66 | 782.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 772.70 | 773.57 | 781.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 773.50 | 779.45 | 782.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:00:00 | 774.10 | 776.80 | 780.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 772.15 | 775.28 | 778.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 777.65 | 773.60 | 777.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 775.90 | 773.60 | 777.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 11:15:00 | 782.30 | 777.34 | 776.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 782.30 | 777.34 | 776.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 783.60 | 778.59 | 777.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 812.70 | 819.23 | 813.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 812.70 | 819.23 | 813.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 812.70 | 819.23 | 813.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 812.70 | 819.23 | 813.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 812.80 | 817.95 | 813.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 811.35 | 817.95 | 813.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 817.10 | 817.78 | 814.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:30:00 | 819.00 | 817.82 | 814.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:30:00 | 819.35 | 818.26 | 815.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 820.10 | 816.69 | 816.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 801.10 | 813.57 | 814.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 801.10 | 813.57 | 814.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 792.00 | 809.26 | 812.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 800.85 | 798.46 | 804.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 800.85 | 798.46 | 804.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 806.65 | 800.35 | 804.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:00:00 | 806.65 | 800.35 | 804.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 803.25 | 800.93 | 803.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 800.90 | 800.93 | 803.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 15:15:00 | 800.90 | 801.52 | 803.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 808.10 | 802.74 | 804.07 | SL hit (close>static) qty=1.00 sl=806.70 alert=retest2 |

### Cycle 146 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 814.00 | 802.47 | 801.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 819.35 | 807.98 | 804.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 804.60 | 810.27 | 806.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 804.60 | 810.27 | 806.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 804.60 | 810.27 | 806.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 804.60 | 810.27 | 806.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 797.10 | 807.64 | 805.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 797.10 | 807.64 | 805.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 793.10 | 802.71 | 803.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 790.20 | 800.21 | 802.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 796.00 | 788.58 | 792.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 796.00 | 788.58 | 792.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 796.00 | 788.58 | 792.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 796.00 | 788.58 | 792.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 795.30 | 789.92 | 792.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 790.15 | 790.54 | 792.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 808.80 | 789.54 | 788.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 808.80 | 789.54 | 788.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 815.45 | 804.18 | 797.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 826.00 | 826.37 | 817.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 825.45 | 826.37 | 817.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 850.80 | 858.13 | 854.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 850.80 | 858.13 | 854.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 852.90 | 857.08 | 854.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 850.00 | 857.08 | 854.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 854.80 | 856.28 | 854.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 854.80 | 856.28 | 854.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 851.00 | 855.22 | 853.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 851.00 | 855.22 | 853.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 854.15 | 855.01 | 853.90 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 847.00 | 852.71 | 853.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 844.25 | 851.02 | 852.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 854.70 | 848.39 | 849.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 854.70 | 848.39 | 849.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 854.70 | 848.39 | 849.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 857.45 | 848.39 | 849.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 855.95 | 849.91 | 850.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 855.95 | 849.91 | 850.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 855.30 | 850.98 | 850.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 858.85 | 852.56 | 851.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 866.80 | 867.25 | 862.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 866.80 | 867.25 | 862.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 866.80 | 867.25 | 862.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 866.80 | 867.25 | 862.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 950.15 | 954.07 | 950.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 948.95 | 954.07 | 950.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 948.85 | 953.02 | 950.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:15:00 | 948.80 | 953.02 | 950.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 954.55 | 953.33 | 950.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 958.85 | 954.66 | 952.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:45:00 | 957.95 | 956.56 | 953.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:45:00 | 956.80 | 956.56 | 954.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 956.90 | 956.38 | 954.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 957.20 | 956.54 | 954.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 957.20 | 956.54 | 954.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 954.70 | 956.17 | 954.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 953.20 | 956.17 | 954.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 957.20 | 956.38 | 954.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:45:00 | 959.85 | 956.63 | 955.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 966.75 | 957.72 | 956.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:30:00 | 960.05 | 961.58 | 959.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 11:15:00 | 952.00 | 959.67 | 959.21 | SL hit (close<static) qty=1.00 sl=953.90 alert=retest2 |

### Cycle 151 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 949.65 | 957.66 | 958.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 14:15:00 | 948.45 | 954.68 | 956.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 963.00 | 954.18 | 955.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 11:15:00 | 963.00 | 954.18 | 955.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 963.00 | 954.18 | 955.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 963.00 | 954.18 | 955.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 963.90 | 956.12 | 956.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 965.20 | 956.12 | 956.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 962.30 | 957.36 | 956.90 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 944.50 | 955.18 | 956.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 942.10 | 950.89 | 954.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 942.50 | 939.89 | 944.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 941.90 | 939.89 | 944.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 946.15 | 941.14 | 944.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 946.15 | 941.14 | 944.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 948.50 | 942.61 | 944.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 947.40 | 942.61 | 944.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 948.50 | 946.24 | 946.10 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 942.50 | 945.62 | 946.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 937.00 | 943.90 | 945.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 931.50 | 930.38 | 934.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:45:00 | 929.55 | 930.38 | 934.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 941.60 | 932.62 | 935.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 941.60 | 932.62 | 935.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 937.50 | 933.60 | 935.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 934.30 | 933.60 | 935.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 936.65 | 934.52 | 935.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 934.00 | 935.42 | 935.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 939.90 | 936.32 | 936.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 939.90 | 936.32 | 936.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 944.75 | 938.00 | 937.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 939.95 | 940.37 | 938.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 939.80 | 940.37 | 938.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 938.20 | 939.94 | 938.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 944.05 | 939.94 | 938.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 958.40 | 963.99 | 964.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 958.40 | 963.99 | 964.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 950.35 | 958.15 | 961.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 943.65 | 943.48 | 948.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 941.35 | 942.83 | 945.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 941.35 | 942.83 | 945.81 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 950.20 | 946.61 | 946.45 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 934.45 | 944.41 | 945.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 928.10 | 938.34 | 942.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 922.50 | 918.79 | 923.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 922.50 | 918.79 | 923.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 922.50 | 918.79 | 923.59 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 928.65 | 924.43 | 924.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 930.45 | 925.64 | 924.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 930.80 | 931.01 | 928.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:15:00 | 930.10 | 931.01 | 928.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 928.95 | 930.60 | 928.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 928.95 | 930.60 | 928.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 927.00 | 929.88 | 928.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 929.90 | 929.88 | 928.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 923.40 | 928.58 | 928.27 | SL hit (close<static) qty=1.00 sl=927.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 924.35 | 927.74 | 927.91 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 928.95 | 926.67 | 926.45 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 924.25 | 926.30 | 926.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 923.50 | 925.74 | 926.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 921.85 | 921.74 | 923.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 13:45:00 | 922.45 | 921.74 | 923.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 923.00 | 921.99 | 923.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 923.45 | 921.99 | 923.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 923.00 | 922.19 | 923.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 924.00 | 922.19 | 923.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 922.65 | 922.29 | 923.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 920.00 | 921.83 | 922.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 920.70 | 920.31 | 921.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 904.70 | 900.38 | 900.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 904.70 | 900.38 | 900.11 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 889.45 | 899.57 | 900.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 888.15 | 892.98 | 895.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 893.85 | 889.27 | 892.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 893.85 | 889.27 | 892.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 893.85 | 889.27 | 892.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 893.85 | 889.27 | 892.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 896.00 | 890.61 | 892.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 896.00 | 890.61 | 892.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 893.00 | 890.75 | 892.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 893.00 | 890.75 | 892.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 894.90 | 891.58 | 892.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 894.85 | 891.58 | 892.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 896.55 | 892.58 | 892.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 895.20 | 892.58 | 892.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 896.50 | 893.36 | 893.15 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 892.95 | 893.28 | 893.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 888.75 | 892.13 | 892.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 10:15:00 | 893.20 | 892.35 | 892.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 893.20 | 892.35 | 892.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 893.20 | 892.35 | 892.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 893.20 | 892.35 | 892.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 11:15:00 | 898.05 | 893.49 | 893.26 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 890.40 | 892.98 | 893.08 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 895.00 | 893.29 | 893.20 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 890.00 | 892.64 | 892.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 884.85 | 891.08 | 892.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 885.95 | 885.87 | 889.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 885.95 | 885.87 | 889.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 172 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 916.05 | 891.95 | 891.23 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 885.25 | 905.49 | 906.95 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 896.20 | 894.93 | 894.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 14:15:00 | 899.40 | 895.82 | 895.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 899.90 | 900.21 | 898.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:00:00 | 899.90 | 900.21 | 898.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 900.05 | 900.18 | 898.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 899.35 | 900.18 | 898.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 895.75 | 899.29 | 897.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 897.15 | 899.29 | 897.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 893.80 | 898.19 | 897.60 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 892.30 | 897.01 | 897.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 888.40 | 894.17 | 895.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 890.85 | 890.69 | 892.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:45:00 | 891.20 | 890.69 | 892.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 894.15 | 891.69 | 893.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 894.00 | 891.69 | 893.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 894.85 | 892.32 | 893.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 888.80 | 893.05 | 893.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 869.35 | 865.61 | 865.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 869.35 | 865.61 | 865.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 887.00 | 872.36 | 869.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 875.50 | 876.83 | 872.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:00:00 | 875.50 | 876.83 | 872.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 866.25 | 875.33 | 873.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 866.25 | 875.33 | 873.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 870.20 | 874.31 | 872.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 871.20 | 872.78 | 872.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 875.20 | 875.93 | 875.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 875.20 | 875.93 | 875.94 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 880.00 | 876.43 | 876.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 883.00 | 879.10 | 877.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 883.10 | 883.40 | 881.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 13:15:00 | 883.10 | 883.40 | 881.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 883.10 | 883.40 | 881.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 883.10 | 883.40 | 881.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 883.70 | 884.29 | 882.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 883.70 | 884.29 | 882.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 884.90 | 884.41 | 882.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 883.00 | 884.41 | 882.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 902.10 | 904.13 | 899.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 902.10 | 904.13 | 899.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 898.90 | 902.42 | 899.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 897.75 | 902.42 | 899.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 902.75 | 902.49 | 899.49 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 896.85 | 898.80 | 898.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 895.20 | 897.89 | 898.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 897.00 | 896.02 | 897.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 10:15:00 | 897.00 | 896.02 | 897.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 897.00 | 896.02 | 897.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 897.00 | 896.02 | 897.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 895.25 | 895.87 | 897.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 893.40 | 895.25 | 896.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:30:00 | 891.25 | 893.60 | 895.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 895.65 | 887.26 | 887.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 895.65 | 887.26 | 887.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 901.65 | 891.33 | 889.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 897.60 | 898.67 | 895.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 11:00:00 | 897.60 | 898.67 | 895.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 898.45 | 899.05 | 895.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 898.45 | 899.05 | 895.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 907.65 | 904.27 | 901.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 909.05 | 904.27 | 901.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:00:00 | 907.85 | 906.62 | 903.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 917.20 | 906.75 | 904.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 892.95 | 905.70 | 906.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 892.95 | 905.70 | 906.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 889.70 | 896.58 | 900.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 896.70 | 895.59 | 898.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 902.60 | 896.99 | 898.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 902.60 | 896.99 | 898.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 902.60 | 896.99 | 898.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 903.25 | 898.24 | 899.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 902.55 | 898.24 | 899.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 901.70 | 899.84 | 899.67 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 897.35 | 899.37 | 899.48 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 901.20 | 899.66 | 899.57 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 898.60 | 899.50 | 899.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 897.00 | 899.00 | 899.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 901.00 | 899.27 | 899.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 901.00 | 899.27 | 899.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 901.00 | 899.27 | 899.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 901.00 | 899.27 | 899.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 899.15 | 899.25 | 899.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:30:00 | 902.95 | 899.25 | 899.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 11:15:00 | 901.00 | 899.60 | 899.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 901.45 | 900.28 | 899.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 12:15:00 | 899.10 | 901.08 | 900.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 12:15:00 | 899.10 | 901.08 | 900.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 899.10 | 901.08 | 900.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 899.10 | 901.08 | 900.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 899.45 | 900.76 | 900.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 899.00 | 900.76 | 900.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 897.75 | 900.15 | 900.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 896.85 | 899.26 | 899.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 15:15:00 | 896.05 | 895.66 | 897.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 892.40 | 895.66 | 897.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 889.50 | 894.43 | 896.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:00:00 | 886.55 | 892.14 | 895.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:30:00 | 885.55 | 890.91 | 894.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 903.85 | 894.21 | 893.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 903.85 | 894.21 | 893.46 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 890.60 | 895.12 | 895.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 889.45 | 893.98 | 894.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 898.80 | 893.42 | 894.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 898.80 | 893.42 | 894.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 898.80 | 893.42 | 894.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 898.80 | 893.42 | 894.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 894.85 | 893.71 | 894.24 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 896.70 | 894.68 | 894.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 913.55 | 899.02 | 896.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 900.20 | 901.82 | 899.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 14:00:00 | 900.20 | 901.82 | 899.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 900.15 | 901.49 | 899.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 900.15 | 901.49 | 899.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 903.10 | 901.81 | 899.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 910.55 | 901.81 | 899.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 906.60 | 902.77 | 900.16 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 900.90 | 902.15 | 902.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 898.70 | 901.20 | 901.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 900.25 | 897.38 | 899.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 900.25 | 897.38 | 899.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 900.25 | 897.38 | 899.09 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 911.20 | 902.17 | 901.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 919.20 | 905.57 | 902.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 912.45 | 913.55 | 908.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 13:00:00 | 912.45 | 913.55 | 908.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 909.20 | 912.68 | 908.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 909.20 | 912.68 | 908.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 905.30 | 911.21 | 908.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 904.20 | 911.21 | 908.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 907.40 | 910.45 | 908.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 898.90 | 910.45 | 908.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 899.15 | 905.95 | 906.64 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 924.40 | 904.18 | 903.70 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 900.00 | 906.66 | 907.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 898.45 | 903.94 | 905.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 902.20 | 902.09 | 903.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 902.20 | 902.09 | 903.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 902.20 | 902.09 | 903.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:15:00 | 900.00 | 902.09 | 903.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 900.00 | 901.67 | 903.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 904.70 | 901.67 | 903.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 906.05 | 902.55 | 903.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 899.55 | 903.16 | 903.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 899.95 | 902.52 | 903.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:00:00 | 899.45 | 901.02 | 902.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 900.00 | 902.01 | 902.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 903.00 | 902.21 | 902.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 903.55 | 902.21 | 902.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 901.75 | 902.11 | 902.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:15:00 | 905.25 | 902.11 | 902.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 904.10 | 902.51 | 902.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 905.75 | 902.51 | 902.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-14 13:15:00 | 907.40 | 903.49 | 903.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 13:15:00 | 907.40 | 903.49 | 903.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 909.40 | 904.67 | 903.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 912.35 | 912.69 | 909.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 912.35 | 912.69 | 909.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 911.10 | 912.37 | 909.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 908.85 | 912.37 | 909.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 920.20 | 915.74 | 912.63 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 908.55 | 912.24 | 912.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 907.30 | 911.25 | 912.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 896.50 | 896.28 | 900.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:00:00 | 896.50 | 896.28 | 900.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 899.20 | 897.19 | 899.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 899.20 | 897.19 | 899.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 897.55 | 897.26 | 899.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 895.85 | 897.61 | 898.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 895.90 | 896.79 | 898.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 896.25 | 897.27 | 898.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:00:00 | 895.85 | 896.99 | 898.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 900.00 | 896.92 | 897.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:30:00 | 901.30 | 896.92 | 897.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 899.50 | 897.43 | 897.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 894.25 | 897.43 | 897.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 898.95 | 897.96 | 898.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 898.70 | 897.96 | 898.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 895.90 | 897.55 | 897.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 895.00 | 897.08 | 897.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 895.20 | 897.08 | 897.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 894.05 | 895.46 | 896.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 851.06 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 851.10 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 851.44 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 851.06 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 850.25 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 850.44 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 849.35 | 859.37 | 866.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 859.50 | 858.65 | 864.48 | SL hit (close>ema200) qty=0.50 sl=858.65 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 862.90 | 861.21 | 861.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 864.25 | 861.82 | 861.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 860.95 | 863.81 | 862.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 860.95 | 863.81 | 862.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 860.95 | 863.81 | 862.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 859.80 | 863.81 | 862.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 857.85 | 862.62 | 862.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 857.85 | 862.62 | 862.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 859.20 | 861.94 | 862.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 856.10 | 859.55 | 860.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 856.40 | 855.59 | 857.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 856.40 | 855.59 | 857.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 856.40 | 855.59 | 857.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 853.25 | 854.55 | 856.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 853.00 | 849.42 | 849.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 853.00 | 849.42 | 849.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 856.00 | 850.73 | 849.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 855.30 | 855.44 | 853.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:30:00 | 854.85 | 855.44 | 853.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 856.80 | 855.82 | 854.38 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 848.20 | 853.01 | 853.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 847.50 | 851.32 | 852.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 850.45 | 850.07 | 851.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 850.45 | 850.07 | 851.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 846.60 | 849.38 | 851.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 845.55 | 848.62 | 850.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 844.35 | 847.75 | 850.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 844.55 | 847.40 | 849.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 843.65 | 847.12 | 849.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 846.45 | 845.36 | 847.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 852.50 | 848.10 | 848.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 852.50 | 848.10 | 848.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 856.60 | 849.80 | 848.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 854.10 | 854.31 | 852.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 852.40 | 853.93 | 852.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 852.40 | 853.93 | 852.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 852.40 | 853.93 | 852.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 852.60 | 853.66 | 852.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 853.25 | 853.53 | 852.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:30:00 | 854.75 | 853.91 | 852.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 854.25 | 855.85 | 855.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 845.00 | 853.68 | 854.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 845.00 | 853.68 | 854.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 841.00 | 848.20 | 850.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 834.55 | 832.54 | 837.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 10:00:00 | 834.55 | 832.54 | 837.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 831.95 | 831.53 | 834.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 826.65 | 831.00 | 833.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 827.90 | 830.44 | 832.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 827.60 | 829.71 | 831.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 828.25 | 829.24 | 831.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 828.00 | 828.02 | 830.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 830.05 | 828.02 | 830.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 826.45 | 827.70 | 829.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 825.35 | 827.08 | 829.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 821.10 | 813.43 | 813.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 821.10 | 813.43 | 813.40 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 802.90 | 812.93 | 814.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 802.15 | 806.73 | 809.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 808.85 | 807.15 | 809.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 808.85 | 807.15 | 809.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 809.95 | 807.71 | 809.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 815.85 | 807.71 | 809.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 810.85 | 808.34 | 809.96 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 819.85 | 811.78 | 811.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 820.55 | 813.53 | 812.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 815.55 | 818.99 | 817.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 815.55 | 818.99 | 817.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 815.55 | 818.99 | 817.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 814.85 | 818.99 | 817.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 815.60 | 818.31 | 817.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 815.30 | 818.31 | 817.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 818.40 | 818.33 | 817.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:45:00 | 819.60 | 818.42 | 817.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 805.75 | 818.74 | 818.60 | SL hit (close<static) qty=1.00 sl=815.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 803.75 | 815.74 | 817.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 800.00 | 812.60 | 815.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 800.80 | 799.24 | 805.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 800.80 | 799.24 | 805.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 806.00 | 800.59 | 805.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 828.40 | 800.59 | 805.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 830.45 | 806.56 | 808.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 830.40 | 806.56 | 808.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 842.80 | 813.81 | 811.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 895.20 | 847.94 | 837.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 891.55 | 892.16 | 882.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:00:00 | 891.55 | 892.16 | 882.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 866.55 | 886.95 | 883.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 866.55 | 886.95 | 883.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 872.40 | 884.04 | 882.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 875.40 | 882.71 | 881.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 876.05 | 881.38 | 881.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 876.05 | 881.38 | 881.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 14:15:00 | 875.50 | 879.50 | 880.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 10:15:00 | 880.70 | 878.76 | 879.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 880.70 | 878.76 | 879.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 880.70 | 878.76 | 879.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 878.65 | 878.76 | 879.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 877.95 | 878.60 | 879.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 875.55 | 879.66 | 879.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 875.60 | 870.63 | 870.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 875.60 | 870.63 | 870.09 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 863.50 | 871.94 | 872.81 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 883.50 | 873.77 | 872.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 885.00 | 880.58 | 878.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 879.00 | 882.96 | 880.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 879.00 | 882.96 | 880.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 879.00 | 882.96 | 880.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 879.00 | 882.96 | 880.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 874.75 | 881.32 | 880.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 874.75 | 881.32 | 880.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 874.50 | 879.95 | 879.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:45:00 | 875.00 | 879.95 | 879.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 875.95 | 879.15 | 879.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 10:15:00 | 869.70 | 876.56 | 878.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 828.95 | 828.36 | 835.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 828.95 | 828.36 | 835.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 831.50 | 828.19 | 834.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 831.50 | 828.19 | 834.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 832.30 | 829.78 | 833.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 834.60 | 829.78 | 833.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 809.65 | 808.27 | 816.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 14:45:00 | 805.85 | 811.05 | 814.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 765.56 | 779.66 | 790.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 778.05 | 774.77 | 783.43 | SL hit (close>ema200) qty=0.50 sl=774.77 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 787.15 | 782.39 | 782.26 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 772.40 | 781.97 | 782.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 767.00 | 775.03 | 778.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 773.35 | 772.27 | 776.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 773.35 | 772.27 | 776.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 773.35 | 772.27 | 776.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 775.35 | 772.27 | 776.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 772.85 | 772.35 | 775.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 775.00 | 772.35 | 775.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 776.20 | 773.12 | 775.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 776.20 | 773.12 | 775.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 775.00 | 773.50 | 775.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 771.80 | 774.11 | 775.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 757.35 | 774.39 | 775.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 780.50 | 763.46 | 761.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 780.50 | 763.46 | 761.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 783.00 | 767.37 | 763.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 768.45 | 773.95 | 769.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 768.45 | 773.95 | 769.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 768.45 | 773.95 | 769.05 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 745.00 | 763.47 | 765.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 737.10 | 758.19 | 763.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 748.30 | 739.38 | 749.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 748.30 | 739.38 | 749.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 748.30 | 739.38 | 749.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 750.20 | 739.38 | 749.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 749.80 | 741.46 | 749.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 749.80 | 741.46 | 749.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 756.30 | 744.43 | 750.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 756.30 | 744.43 | 750.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 757.00 | 746.94 | 750.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 757.35 | 746.94 | 750.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 746.50 | 747.18 | 750.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 724.55 | 747.27 | 749.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 743.80 | 739.10 | 740.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 15:15:00 | 746.90 | 741.61 | 741.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 746.90 | 741.61 | 741.55 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 737.35 | 740.76 | 741.16 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 791.40 | 750.94 | 745.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 800.80 | 760.92 | 750.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 798.45 | 799.95 | 790.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:30:00 | 796.85 | 799.95 | 790.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 790.20 | 798.14 | 791.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 796.10 | 796.53 | 791.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 827.20 | 832.47 | 832.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 827.20 | 832.47 | 832.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 09:15:00 | 826.45 | 831.26 | 832.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 811.65 | 810.23 | 815.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 811.65 | 810.23 | 815.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 819.70 | 812.47 | 815.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 819.70 | 812.47 | 815.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 820.00 | 813.98 | 815.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 820.00 | 813.98 | 815.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 819.00 | 817.01 | 816.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 822.00 | 818.76 | 818.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 814.50 | 818.36 | 818.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 814.50 | 818.36 | 818.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 814.50 | 818.36 | 818.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 814.50 | 818.36 | 818.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 811.95 | 817.08 | 817.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 801.80 | 813.21 | 815.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 801.70 | 800.97 | 804.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 15:00:00 | 801.70 | 800.97 | 804.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 802.00 | 800.31 | 802.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 804.00 | 800.31 | 802.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 800.25 | 800.29 | 801.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 799.60 | 800.23 | 801.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 806.25 | 801.68 | 802.05 | SL hit (close>static) qty=1.00 sl=802.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 807.05 | 802.76 | 802.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 809.60 | 804.13 | 803.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 804.35 | 808.14 | 806.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 804.35 | 808.14 | 806.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 804.35 | 808.14 | 806.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 804.00 | 808.14 | 806.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 803.60 | 807.23 | 806.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 803.60 | 807.23 | 806.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 801.10 | 804.49 | 804.93 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-25 13:45:00 | 984.25 | 2024-04-29 10:15:00 | 981.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-04-29 10:00:00 | 984.35 | 2024-04-29 10:15:00 | 981.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-05-10 12:30:00 | 907.00 | 2024-05-14 13:15:00 | 924.15 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-05-10 14:00:00 | 907.25 | 2024-05-14 13:15:00 | 924.15 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-05-13 09:15:00 | 905.45 | 2024-05-14 13:15:00 | 924.15 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-05-14 09:30:00 | 907.60 | 2024-05-14 13:15:00 | 924.15 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-05-24 12:00:00 | 1044.35 | 2024-05-28 09:15:00 | 1018.35 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-05-27 10:30:00 | 1051.90 | 2024-05-28 09:15:00 | 1018.35 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-05-27 13:00:00 | 1039.10 | 2024-05-28 09:15:00 | 1018.35 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-05-28 09:15:00 | 1043.00 | 2024-05-28 09:15:00 | 1018.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1008.65 | 2024-06-20 09:15:00 | 1032.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2024-06-13 11:30:00 | 999.95 | 2024-06-20 09:15:00 | 1032.00 | STOP_HIT | 1.00 | 3.21% |
| BUY | retest2 | 2024-06-13 12:30:00 | 1000.40 | 2024-06-20 09:15:00 | 1032.00 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2024-06-13 13:30:00 | 1000.30 | 2024-06-20 09:15:00 | 1032.00 | STOP_HIT | 1.00 | 3.17% |
| SELL | retest2 | 2024-06-28 12:15:00 | 995.75 | 2024-07-04 09:15:00 | 995.00 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-06-28 13:30:00 | 994.00 | 2024-07-04 09:15:00 | 995.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-06-28 14:30:00 | 994.90 | 2024-07-04 09:15:00 | 995.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-07-01 14:15:00 | 997.55 | 2024-07-04 09:15:00 | 995.00 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-07-02 11:30:00 | 988.75 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-07-02 12:00:00 | 987.80 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-07-02 15:00:00 | 985.05 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2024-07-03 09:45:00 | 988.70 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-07-03 11:15:00 | 988.60 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-07-03 12:30:00 | 989.00 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-07-03 13:30:00 | 988.95 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-07-03 14:00:00 | 988.75 | 2024-07-04 10:15:00 | 1018.70 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-07-19 13:45:00 | 1087.50 | 2024-07-23 14:15:00 | 1085.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-07-23 12:30:00 | 1095.00 | 2024-07-23 14:15:00 | 1085.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-07-29 13:30:00 | 1179.30 | 2024-08-02 12:15:00 | 1174.85 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-29 15:15:00 | 1180.10 | 2024-08-02 12:15:00 | 1174.85 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-07-30 13:30:00 | 1180.95 | 2024-08-02 12:15:00 | 1174.85 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-07-31 11:00:00 | 1181.50 | 2024-08-02 12:15:00 | 1174.85 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-08-01 09:15:00 | 1183.60 | 2024-08-02 12:15:00 | 1174.85 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-08-02 10:45:00 | 1185.00 | 2024-08-02 12:15:00 | 1174.85 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1129.25 | 2024-08-08 12:15:00 | 1134.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-08-21 12:15:00 | 1079.45 | 2024-08-23 11:15:00 | 1067.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-08-21 14:15:00 | 1079.60 | 2024-08-23 11:15:00 | 1067.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-08-21 14:45:00 | 1081.10 | 2024-08-23 11:15:00 | 1067.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-08-22 11:45:00 | 1080.10 | 2024-08-23 11:15:00 | 1067.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-09-03 11:00:00 | 1059.35 | 2024-09-05 10:15:00 | 1065.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-09-03 13:30:00 | 1060.10 | 2024-09-05 10:15:00 | 1065.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-09-04 09:15:00 | 1060.00 | 2024-09-05 10:15:00 | 1065.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-09-11 09:30:00 | 1031.60 | 2024-09-13 09:15:00 | 1041.05 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-09-18 12:45:00 | 1023.30 | 2024-09-23 10:15:00 | 1026.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-09-19 09:30:00 | 1022.55 | 2024-09-23 10:15:00 | 1026.30 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-10-11 09:45:00 | 968.40 | 2024-10-11 11:15:00 | 957.05 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-14 13:00:00 | 957.25 | 2024-10-22 12:15:00 | 909.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 10:00:00 | 956.80 | 2024-10-23 09:15:00 | 908.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 13:00:00 | 957.25 | 2024-10-23 10:15:00 | 918.05 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2024-10-15 10:00:00 | 956.80 | 2024-10-23 10:15:00 | 918.05 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-11-11 13:15:00 | 921.30 | 2024-11-12 09:15:00 | 953.50 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-11-19 14:45:00 | 899.00 | 2024-11-25 11:15:00 | 904.45 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-04 09:15:00 | 975.60 | 2024-12-10 09:15:00 | 957.10 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-12-05 11:45:00 | 971.35 | 2024-12-10 09:15:00 | 957.10 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-17 11:30:00 | 926.45 | 2024-12-30 13:15:00 | 880.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:30:00 | 926.45 | 2024-12-30 14:15:00 | 905.25 | STOP_HIT | 0.50 | 2.29% |
| BUY | retest1 | 2025-01-21 09:30:00 | 860.05 | 2025-01-21 10:15:00 | 842.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-01-24 12:30:00 | 827.70 | 2025-01-29 10:15:00 | 831.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-02-18 09:15:00 | 761.55 | 2025-02-19 09:15:00 | 770.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-02-18 10:15:00 | 764.25 | 2025-02-19 09:15:00 | 770.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-02-18 11:00:00 | 763.75 | 2025-02-19 09:15:00 | 770.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-02-18 11:45:00 | 760.00 | 2025-02-19 09:15:00 | 770.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-02-19 13:45:00 | 762.00 | 2025-02-20 10:15:00 | 769.75 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-02-19 14:15:00 | 760.15 | 2025-02-20 10:15:00 | 769.75 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-03-04 11:30:00 | 734.10 | 2025-03-05 09:15:00 | 748.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-04 14:30:00 | 734.15 | 2025-03-05 09:15:00 | 748.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-04 15:00:00 | 733.30 | 2025-03-05 09:15:00 | 748.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-03-12 10:45:00 | 744.40 | 2025-03-18 10:15:00 | 751.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-03-13 10:30:00 | 746.80 | 2025-03-18 10:15:00 | 751.70 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-03-13 11:00:00 | 745.60 | 2025-03-18 10:15:00 | 751.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-03-17 09:45:00 | 745.90 | 2025-03-18 10:15:00 | 751.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-03-26 09:15:00 | 797.40 | 2025-03-26 13:15:00 | 789.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-04-08 10:30:00 | 772.70 | 2025-04-15 11:15:00 | 782.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-04-09 10:15:00 | 773.50 | 2025-04-15 11:15:00 | 782.30 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-04-09 12:00:00 | 774.10 | 2025-04-15 11:15:00 | 782.30 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-04-09 13:30:00 | 772.15 | 2025-04-15 11:15:00 | 782.30 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-04-11 10:15:00 | 775.90 | 2025-04-15 11:15:00 | 782.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-04-23 12:30:00 | 819.00 | 2025-04-25 09:15:00 | 801.10 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-04-23 13:30:00 | 819.35 | 2025-04-25 09:15:00 | 801.10 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-04-25 09:15:00 | 820.10 | 2025-04-25 09:15:00 | 801.10 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-04-28 14:15:00 | 800.90 | 2025-04-29 09:15:00 | 808.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-28 15:15:00 | 800.90 | 2025-04-29 09:15:00 | 808.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-29 12:15:00 | 801.80 | 2025-05-05 10:15:00 | 808.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-04-29 14:15:00 | 801.00 | 2025-05-05 10:15:00 | 808.35 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-04-30 13:30:00 | 803.50 | 2025-05-05 10:15:00 | 808.35 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-05-02 10:15:00 | 803.75 | 2025-05-05 10:15:00 | 808.35 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-05-08 13:00:00 | 790.15 | 2025-05-12 09:15:00 | 808.80 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-06-05 10:00:00 | 958.85 | 2025-06-10 11:15:00 | 952.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-06-05 12:45:00 | 957.95 | 2025-06-10 11:15:00 | 952.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-05 14:45:00 | 956.80 | 2025-06-10 11:15:00 | 952.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-06-06 09:15:00 | 956.90 | 2025-06-10 12:15:00 | 949.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-06 12:45:00 | 959.85 | 2025-06-10 12:15:00 | 949.65 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-09 09:15:00 | 966.75 | 2025-06-10 12:15:00 | 949.65 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-10 10:30:00 | 960.05 | 2025-06-10 12:15:00 | 949.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-20 12:15:00 | 934.30 | 2025-06-23 09:15:00 | 939.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-20 14:00:00 | 936.65 | 2025-06-23 09:15:00 | 939.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-23 09:15:00 | 934.00 | 2025-06-23 09:15:00 | 939.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-24 09:15:00 | 944.05 | 2025-07-02 13:15:00 | 958.40 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2025-07-18 09:15:00 | 929.90 | 2025-07-18 09:15:00 | 923.40 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-24 11:00:00 | 920.00 | 2025-07-30 11:15:00 | 904.70 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-07-24 14:30:00 | 920.70 | 2025-07-30 11:15:00 | 904.70 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2025-08-26 09:15:00 | 888.80 | 2025-09-03 10:15:00 | 869.35 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-09-05 13:30:00 | 871.20 | 2025-09-12 10:15:00 | 875.20 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-09-25 14:00:00 | 893.40 | 2025-09-29 14:15:00 | 895.65 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-09-25 14:30:00 | 891.25 | 2025-09-29 14:15:00 | 895.65 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-06 10:15:00 | 909.05 | 2025-10-08 10:15:00 | 892.95 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-06 15:00:00 | 907.85 | 2025-10-08 10:15:00 | 892.95 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-10-07 09:15:00 | 917.20 | 2025-10-08 10:15:00 | 892.95 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-10-17 13:00:00 | 886.55 | 2025-10-23 09:15:00 | 903.85 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-17 13:30:00 | 885.55 | 2025-10-23 09:15:00 | 903.85 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-11-12 14:00:00 | 899.55 | 2025-11-14 13:15:00 | 907.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-12 15:00:00 | 899.95 | 2025-11-14 13:15:00 | 907.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-13 11:00:00 | 899.45 | 2025-11-14 13:15:00 | 907.40 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-14 10:00:00 | 900.00 | 2025-11-14 13:15:00 | 907.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-26 14:45:00 | 895.85 | 2025-12-09 09:15:00 | 851.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 09:30:00 | 895.90 | 2025-12-09 09:15:00 | 851.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:15:00 | 896.25 | 2025-12-09 09:15:00 | 851.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 12:00:00 | 895.85 | 2025-12-09 09:15:00 | 851.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 12:30:00 | 895.00 | 2025-12-09 09:15:00 | 850.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 13:00:00 | 895.20 | 2025-12-09 09:15:00 | 850.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 894.05 | 2025-12-09 09:15:00 | 849.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 14:45:00 | 895.85 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-11-27 09:30:00 | 895.90 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-11-27 11:15:00 | 896.25 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2025-11-27 12:00:00 | 895.85 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-11-28 12:30:00 | 895.00 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2025-11-28 13:00:00 | 895.20 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-12-01 10:45:00 | 894.05 | 2025-12-09 11:15:00 | 859.50 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-12-17 10:30:00 | 853.25 | 2025-12-19 15:15:00 | 853.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-29 11:00:00 | 845.55 | 2025-12-31 09:15:00 | 852.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-12-29 11:45:00 | 844.35 | 2025-12-31 09:15:00 | 852.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-29 13:15:00 | 844.55 | 2025-12-31 09:15:00 | 852.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-29 14:15:00 | 843.65 | 2025-12-31 09:15:00 | 852.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-01 14:15:00 | 853.25 | 2026-01-05 13:15:00 | 845.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-02 09:30:00 | 854.75 | 2026-01-05 13:15:00 | 845.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-01-05 12:30:00 | 854.25 | 2026-01-05 13:15:00 | 845.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-13 14:15:00 | 826.65 | 2026-01-22 13:15:00 | 821.10 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2026-01-14 10:00:00 | 827.90 | 2026-01-22 13:15:00 | 821.10 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2026-01-14 12:30:00 | 827.60 | 2026-01-22 13:15:00 | 821.10 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2026-01-14 13:30:00 | 828.25 | 2026-01-22 13:15:00 | 821.10 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2026-01-16 11:45:00 | 825.35 | 2026-01-22 13:15:00 | 821.10 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2026-01-30 13:45:00 | 819.60 | 2026-02-01 12:15:00 | 805.75 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-11 11:30:00 | 875.40 | 2026-02-11 12:15:00 | 876.05 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-02-13 09:15:00 | 875.55 | 2026-02-17 12:15:00 | 875.60 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-03-11 14:45:00 | 805.85 | 2026-03-16 09:15:00 | 765.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:45:00 | 805.85 | 2026-03-16 14:15:00 | 778.05 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2026-03-20 14:30:00 | 771.80 | 2026-03-25 10:15:00 | 780.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-03-23 09:15:00 | 757.35 | 2026-03-25 10:15:00 | 780.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-04-02 09:15:00 | 724.55 | 2026-04-06 15:15:00 | 746.90 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-04-06 14:15:00 | 743.80 | 2026-04-06 15:15:00 | 746.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-04-13 12:15:00 | 796.10 | 2026-04-20 15:15:00 | 827.20 | STOP_HIT | 1.00 | 3.91% |
| SELL | retest2 | 2026-05-06 12:45:00 | 799.60 | 2026-05-06 14:15:00 | 806.25 | STOP_HIT | 1.00 | -0.83% |
