# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1277.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 250 |
| ALERT1 | 158 |
| ALERT2 | 157 |
| ALERT2_SKIP | 115 |
| ALERT3 | 367 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 133 |
| PARTIAL | 15 |
| TARGET_HIT | 12 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 111
- **Target hits / Stop hits / Partials:** 12 / 121 / 15
- **Avg / median % per leg:** 0.20% / -0.91%
- **Sum % (uncompounded):** 29.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 5 | 7.0% | 3 | 68 | 0 | -0.96% | -68.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 71 | 5 | 7.0% | 3 | 68 | 0 | -0.96% | -68.2% |
| SELL (all) | 77 | 32 | 41.6% | 9 | 53 | 15 | 1.27% | 98.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 77 | 32 | 41.6% | 9 | 53 | 15 | 1.27% | 98.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 148 | 37 | 25.0% | 12 | 121 | 15 | 0.20% | 29.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 522.40 | 519.07 | 518.85 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 14:15:00 | 516.55 | 518.52 | 518.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 11:15:00 | 514.65 | 517.10 | 517.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 517.30 | 516.48 | 517.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 517.30 | 516.48 | 517.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 517.30 | 516.48 | 517.38 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 12:15:00 | 524.00 | 513.08 | 512.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 12:15:00 | 533.50 | 524.92 | 519.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 535.80 | 537.18 | 531.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 535.80 | 537.18 | 531.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 535.80 | 537.18 | 531.93 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 09:15:00 | 515.80 | 531.53 | 531.60 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 531.70 | 527.12 | 526.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 11:15:00 | 533.90 | 528.48 | 527.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 14:15:00 | 525.95 | 529.92 | 528.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 14:15:00 | 525.95 | 529.92 | 528.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 525.95 | 529.92 | 528.46 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 10:15:00 | 531.50 | 533.61 | 533.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 15:15:00 | 529.90 | 532.98 | 533.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 14:15:00 | 535.50 | 529.19 | 530.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 14:15:00 | 535.50 | 529.19 | 530.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 535.50 | 529.19 | 530.19 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 537.50 | 531.83 | 531.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 10:15:00 | 541.25 | 533.72 | 532.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 11:15:00 | 538.00 | 541.00 | 537.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 11:15:00 | 538.00 | 541.00 | 537.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 538.00 | 541.00 | 537.84 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 12:15:00 | 535.75 | 537.33 | 537.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 531.65 | 534.66 | 535.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 12:15:00 | 527.50 | 523.55 | 526.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 12:15:00 | 527.50 | 523.55 | 526.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 527.50 | 523.55 | 526.69 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 531.60 | 527.53 | 527.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 538.35 | 529.70 | 528.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 10:15:00 | 530.40 | 531.62 | 529.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 10:15:00 | 530.40 | 531.62 | 529.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 530.40 | 531.62 | 529.68 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 12:15:00 | 550.45 | 551.67 | 551.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 13:15:00 | 545.55 | 550.45 | 551.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 10:15:00 | 546.90 | 546.37 | 548.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 11:15:00 | 551.50 | 547.40 | 548.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 551.50 | 547.40 | 548.98 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 555.35 | 550.42 | 550.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 14:15:00 | 557.70 | 555.18 | 553.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 11:15:00 | 554.90 | 556.75 | 554.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 11:15:00 | 554.90 | 556.75 | 554.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 554.90 | 556.75 | 554.79 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 556.30 | 562.31 | 562.76 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 12:15:00 | 564.45 | 563.13 | 563.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 568.05 | 564.46 | 563.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 11:15:00 | 617.30 | 619.06 | 610.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 14:15:00 | 629.65 | 631.12 | 626.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 629.65 | 631.12 | 626.88 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 10:15:00 | 615.90 | 628.51 | 629.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 12:15:00 | 595.05 | 619.86 | 625.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 09:15:00 | 628.00 | 614.86 | 620.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 628.00 | 614.86 | 620.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 628.00 | 614.86 | 620.47 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 13:15:00 | 630.75 | 623.58 | 623.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 09:15:00 | 638.00 | 628.72 | 625.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 14:15:00 | 624.00 | 629.53 | 627.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 14:15:00 | 624.00 | 629.53 | 627.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 624.00 | 629.53 | 627.63 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 622.10 | 626.24 | 626.45 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 12:15:00 | 631.35 | 626.82 | 626.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 13:15:00 | 637.40 | 628.94 | 627.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 12:15:00 | 643.70 | 644.33 | 639.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 13:15:00 | 643.70 | 644.21 | 639.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 643.70 | 644.21 | 639.81 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 631.10 | 637.61 | 637.84 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 13:15:00 | 642.70 | 638.51 | 638.14 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 15:15:00 | 636.00 | 638.17 | 638.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 11:15:00 | 629.75 | 635.92 | 637.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 14:15:00 | 636.25 | 634.53 | 636.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 14:15:00 | 636.25 | 634.53 | 636.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 636.25 | 634.53 | 636.16 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 638.00 | 635.43 | 635.28 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 11:15:00 | 634.55 | 635.23 | 635.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 12:15:00 | 631.95 | 634.58 | 634.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 15:15:00 | 636.95 | 634.55 | 634.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 15:15:00 | 636.95 | 634.55 | 634.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 636.95 | 634.55 | 634.80 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 638.85 | 635.41 | 635.16 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 12:15:00 | 628.00 | 635.79 | 635.98 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 636.15 | 631.99 | 631.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 648.00 | 637.24 | 634.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 645.00 | 645.53 | 640.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 15:15:00 | 636.00 | 643.62 | 640.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 636.00 | 643.62 | 640.46 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 10:15:00 | 668.35 | 672.33 | 672.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 11:15:00 | 665.20 | 670.90 | 671.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 669.85 | 668.67 | 670.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 669.85 | 668.67 | 670.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 669.85 | 668.67 | 670.19 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 14:15:00 | 656.35 | 652.37 | 652.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 669.00 | 656.35 | 654.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 663.95 | 663.99 | 659.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 15:15:00 | 652.30 | 661.65 | 658.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 652.30 | 661.65 | 658.85 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 645.55 | 656.29 | 657.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 11:15:00 | 639.00 | 642.83 | 648.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 628.95 | 626.82 | 632.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 633.35 | 628.13 | 632.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 633.35 | 628.13 | 632.44 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 634.20 | 631.62 | 631.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 11:15:00 | 636.20 | 632.53 | 632.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 12:15:00 | 630.65 | 632.16 | 631.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 12:15:00 | 630.65 | 632.16 | 631.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 630.65 | 632.16 | 631.89 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 13:15:00 | 628.60 | 631.45 | 631.60 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 10:15:00 | 632.60 | 631.73 | 631.61 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 626.00 | 630.86 | 631.42 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 636.85 | 632.58 | 632.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 09:15:00 | 657.25 | 637.73 | 634.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 15:15:00 | 644.35 | 646.50 | 641.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 15:15:00 | 644.35 | 646.50 | 641.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 644.35 | 646.50 | 641.31 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 11:15:00 | 641.00 | 646.84 | 647.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 12:15:00 | 640.85 | 645.64 | 646.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 09:15:00 | 644.90 | 641.92 | 644.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 644.90 | 641.92 | 644.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 644.90 | 641.92 | 644.13 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 649.35 | 642.61 | 642.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 15:15:00 | 651.40 | 646.38 | 644.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 654.00 | 654.03 | 650.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 684.95 | 686.95 | 680.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 684.95 | 686.95 | 680.22 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 10:15:00 | 683.25 | 697.39 | 698.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 673.50 | 692.61 | 696.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 15:15:00 | 657.95 | 653.15 | 667.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 14:15:00 | 650.95 | 644.02 | 649.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 650.95 | 644.02 | 649.87 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 673.00 | 654.21 | 653.36 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 13:15:00 | 657.80 | 659.60 | 659.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 15:15:00 | 650.00 | 657.34 | 658.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 13:15:00 | 608.75 | 608.45 | 615.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 11:15:00 | 614.95 | 609.27 | 613.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 614.95 | 609.27 | 613.25 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 10:15:00 | 622.95 | 615.77 | 615.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 14:15:00 | 629.90 | 620.71 | 617.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 09:15:00 | 641.40 | 651.16 | 644.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 641.40 | 651.16 | 644.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 641.40 | 651.16 | 644.20 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 642.95 | 652.41 | 653.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 10:15:00 | 642.05 | 650.34 | 652.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 12:15:00 | 642.00 | 641.71 | 645.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 15:15:00 | 642.95 | 642.29 | 644.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 642.95 | 642.29 | 644.88 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 652.65 | 640.56 | 639.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 13:15:00 | 657.55 | 645.62 | 641.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 15:15:00 | 663.20 | 663.95 | 656.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 13:15:00 | 692.75 | 700.83 | 699.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 692.75 | 700.83 | 699.21 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 686.00 | 695.83 | 697.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 10:15:00 | 674.65 | 682.31 | 685.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 682.45 | 679.81 | 683.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 14:15:00 | 682.45 | 679.81 | 683.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 682.45 | 679.81 | 683.28 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 11:15:00 | 648.95 | 643.25 | 642.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 09:15:00 | 655.60 | 648.41 | 645.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 15:15:00 | 699.90 | 699.96 | 681.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 702.40 | 708.56 | 697.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 702.40 | 708.56 | 697.99 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 15:15:00 | 734.00 | 734.92 | 734.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 731.35 | 734.21 | 734.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 11:15:00 | 734.00 | 733.94 | 734.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 11:15:00 | 734.00 | 733.94 | 734.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 734.00 | 733.94 | 734.45 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 742.25 | 734.93 | 734.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 14:15:00 | 743.75 | 739.16 | 737.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 15:15:00 | 739.00 | 739.13 | 737.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 10:15:00 | 738.50 | 739.77 | 738.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 10:15:00 | 738.50 | 739.77 | 738.07 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 726.50 | 736.06 | 736.95 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 741.65 | 736.48 | 736.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 13:15:00 | 745.70 | 739.29 | 737.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 740.50 | 743.51 | 740.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 11:15:00 | 740.50 | 743.51 | 740.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 740.50 | 743.51 | 740.88 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 731.80 | 739.48 | 739.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 14:15:00 | 718.70 | 731.46 | 735.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 740.25 | 731.76 | 734.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 740.25 | 731.76 | 734.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 740.25 | 731.76 | 734.83 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 733.20 | 727.78 | 727.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 738.35 | 729.89 | 728.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 12:15:00 | 862.25 | 876.55 | 860.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 13:15:00 | 852.85 | 871.81 | 859.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 852.85 | 871.81 | 859.50 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 11:15:00 | 856.80 | 862.29 | 862.37 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 13:15:00 | 871.00 | 864.00 | 863.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 15:15:00 | 872.95 | 865.55 | 863.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 11:15:00 | 866.95 | 867.05 | 865.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 11:15:00 | 866.95 | 867.05 | 865.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 866.95 | 867.05 | 865.13 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 12:15:00 | 944.90 | 952.35 | 952.51 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 13:15:00 | 954.30 | 952.74 | 952.67 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 948.00 | 951.79 | 952.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 15:15:00 | 945.00 | 950.43 | 951.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 938.10 | 932.36 | 939.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 938.10 | 932.36 | 939.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 938.10 | 932.36 | 939.11 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 952.10 | 934.41 | 932.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 985.05 | 947.69 | 939.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 15:15:00 | 983.00 | 986.43 | 972.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 14:15:00 | 982.65 | 988.82 | 980.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 982.65 | 988.82 | 980.24 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 954.75 | 981.99 | 982.83 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 982.80 | 972.54 | 972.50 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 964.80 | 972.34 | 972.50 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 10:15:00 | 1006.70 | 979.21 | 975.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 09:15:00 | 1018.00 | 997.34 | 987.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 10:15:00 | 993.65 | 999.86 | 994.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 10:15:00 | 993.65 | 999.86 | 994.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 993.65 | 999.86 | 994.51 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 15:15:00 | 987.00 | 991.59 | 991.97 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 1007.90 | 995.29 | 993.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 11:15:00 | 1026.95 | 1001.62 | 996.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 09:15:00 | 1069.20 | 1093.90 | 1065.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 1069.20 | 1093.90 | 1065.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 1069.20 | 1093.90 | 1065.04 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 10:15:00 | 1023.80 | 1056.00 | 1057.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 1007.70 | 1046.34 | 1053.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 1032.00 | 1018.88 | 1031.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 1032.00 | 1018.88 | 1031.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 1032.00 | 1018.88 | 1031.86 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 1036.00 | 1032.10 | 1031.82 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 1030.15 | 1031.89 | 1031.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 996.15 | 1014.23 | 1021.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 1028.40 | 1012.85 | 1019.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 1028.40 | 1012.85 | 1019.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1028.40 | 1012.85 | 1019.02 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 14:15:00 | 1081.80 | 1031.46 | 1026.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 1123.00 | 1071.90 | 1050.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 12:15:00 | 1083.50 | 1085.06 | 1069.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 13:15:00 | 1069.45 | 1081.94 | 1069.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 13:15:00 | 1069.45 | 1081.94 | 1069.30 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 09:15:00 | 1062.00 | 1066.86 | 1067.09 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 1069.00 | 1067.37 | 1067.29 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 12:15:00 | 1061.60 | 1066.22 | 1066.77 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 1080.00 | 1068.36 | 1067.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 1100.00 | 1080.31 | 1074.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 15:15:00 | 1119.00 | 1128.48 | 1118.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 15:15:00 | 1119.00 | 1128.48 | 1118.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 15:15:00 | 1119.00 | 1128.48 | 1118.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1191.65 | 1185.89 | 1171.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 1145.30 | 1172.27 | 1172.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 1145.30 | 1172.27 | 1172.85 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 1179.75 | 1162.51 | 1162.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 13:15:00 | 1191.90 | 1169.77 | 1165.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 11:15:00 | 1192.05 | 1194.64 | 1185.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-22 11:30:00 | 1191.00 | 1194.64 | 1185.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 1207.95 | 1197.30 | 1187.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:45:00 | 1201.00 | 1197.30 | 1187.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 1206.70 | 1201.16 | 1191.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 1206.70 | 1201.16 | 1191.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1190.50 | 1199.18 | 1192.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:15:00 | 1190.00 | 1199.18 | 1192.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 1185.00 | 1196.34 | 1192.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:00:00 | 1185.00 | 1196.34 | 1192.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 1187.00 | 1194.47 | 1191.61 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 1182.85 | 1189.89 | 1190.05 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 10:15:00 | 1198.15 | 1184.79 | 1183.19 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 14:15:00 | 1170.00 | 1180.10 | 1181.40 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1240.50 | 1190.88 | 1186.00 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 1211.05 | 1225.43 | 1225.47 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 10:15:00 | 1238.40 | 1224.17 | 1224.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 15:15:00 | 1246.95 | 1236.84 | 1231.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 1215.90 | 1232.65 | 1229.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 1215.90 | 1232.65 | 1229.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1215.90 | 1232.65 | 1229.77 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 1214.55 | 1226.01 | 1227.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 1207.05 | 1220.62 | 1224.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 1167.75 | 1164.83 | 1177.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 1167.75 | 1164.83 | 1177.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1167.75 | 1164.83 | 1177.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 1181.15 | 1164.83 | 1177.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 1165.00 | 1161.41 | 1169.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:30:00 | 1182.75 | 1161.41 | 1169.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 1171.95 | 1163.52 | 1169.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:15:00 | 1184.70 | 1163.52 | 1169.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 1179.90 | 1166.79 | 1170.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 1184.80 | 1166.79 | 1170.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 15:15:00 | 1185.00 | 1174.14 | 1173.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1199.40 | 1179.19 | 1175.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 1196.00 | 1199.30 | 1193.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 12:00:00 | 1196.00 | 1199.30 | 1193.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 1195.00 | 1198.44 | 1194.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 1283.90 | 1198.44 | 1194.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 1216.70 | 1226.23 | 1226.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1216.70 | 1226.23 | 1226.96 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 1232.90 | 1227.56 | 1227.50 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 1222.00 | 1226.74 | 1227.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 1212.05 | 1223.90 | 1225.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 1234.00 | 1223.21 | 1224.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 1234.00 | 1223.21 | 1224.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1234.00 | 1223.21 | 1224.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:45:00 | 1238.25 | 1223.21 | 1224.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1212.50 | 1221.07 | 1223.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 1212.05 | 1218.75 | 1222.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:00:00 | 1209.45 | 1218.75 | 1222.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 13:45:00 | 1208.10 | 1215.52 | 1220.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:00:00 | 1210.20 | 1212.99 | 1217.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 1218.95 | 1214.50 | 1217.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:15:00 | 1219.35 | 1214.50 | 1217.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1213.15 | 1214.23 | 1216.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 1210.25 | 1214.31 | 1216.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:30:00 | 1209.05 | 1210.80 | 1214.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1236.30 | 1198.26 | 1198.65 | SL hit (close>static) qty=1.00 sl=1235.85 alert=retest2 |

### Cycle 83 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 1211.60 | 1200.93 | 1199.83 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1105.00 | 1186.12 | 1195.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 997.95 | 1148.49 | 1177.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 1097.40 | 1093.81 | 1123.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 1097.40 | 1093.81 | 1123.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1170.35 | 1111.99 | 1124.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 1170.35 | 1111.99 | 1124.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1175.75 | 1124.74 | 1129.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 1175.75 | 1124.74 | 1129.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1168.70 | 1133.53 | 1132.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 1186.40 | 1150.74 | 1141.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1269.65 | 1271.25 | 1242.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 14:15:00 | 1237.95 | 1260.47 | 1248.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 1237.95 | 1260.47 | 1248.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 1237.95 | 1260.47 | 1248.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1234.90 | 1255.36 | 1247.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 1235.10 | 1255.36 | 1247.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1248.50 | 1252.99 | 1247.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 1247.95 | 1252.99 | 1247.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1247.65 | 1251.92 | 1247.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 1246.25 | 1251.92 | 1247.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1238.50 | 1249.24 | 1246.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 1234.85 | 1249.24 | 1246.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1246.00 | 1252.10 | 1249.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 1246.00 | 1252.10 | 1249.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1238.95 | 1249.47 | 1248.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:45:00 | 1239.40 | 1249.47 | 1248.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 1234.00 | 1246.37 | 1247.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 12:15:00 | 1230.10 | 1243.12 | 1245.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 1158.95 | 1158.15 | 1172.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:45:00 | 1158.80 | 1158.15 | 1172.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1157.35 | 1157.19 | 1169.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 1156.95 | 1157.19 | 1169.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1163.10 | 1156.95 | 1163.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1160.60 | 1156.95 | 1163.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1171.10 | 1159.78 | 1164.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 1171.10 | 1159.78 | 1164.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1173.40 | 1162.51 | 1165.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 1173.40 | 1162.51 | 1165.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1168.60 | 1163.73 | 1165.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:30:00 | 1169.10 | 1163.73 | 1165.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 1173.05 | 1166.87 | 1166.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 1191.00 | 1174.02 | 1170.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 1166.40 | 1172.97 | 1170.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 11:15:00 | 1166.40 | 1172.97 | 1170.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1166.40 | 1172.97 | 1170.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:00:00 | 1166.40 | 1172.97 | 1170.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1176.60 | 1173.69 | 1170.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:30:00 | 1160.25 | 1173.69 | 1170.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1178.90 | 1174.73 | 1171.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:45:00 | 1172.00 | 1174.73 | 1171.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1173.00 | 1177.31 | 1173.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 1174.90 | 1177.31 | 1173.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1180.80 | 1178.00 | 1174.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:15:00 | 1194.10 | 1178.69 | 1175.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 1187.35 | 1182.78 | 1178.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1172.60 | 1180.74 | 1177.83 | SL hit (close<static) qty=1.00 sl=1172.70 alert=retest2 |

### Cycle 88 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 1162.20 | 1174.99 | 1175.60 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 1180.95 | 1176.46 | 1175.97 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 10:15:00 | 1170.05 | 1175.47 | 1175.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 09:15:00 | 1154.65 | 1170.39 | 1173.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 1167.15 | 1161.71 | 1166.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 14:15:00 | 1167.15 | 1161.71 | 1166.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 1167.15 | 1161.71 | 1166.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:30:00 | 1168.75 | 1161.71 | 1166.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 1173.40 | 1164.05 | 1167.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:30:00 | 1167.55 | 1165.24 | 1167.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1179.30 | 1168.05 | 1168.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 1177.00 | 1168.05 | 1168.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 11:15:00 | 1211.55 | 1176.75 | 1172.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 1233.15 | 1208.37 | 1192.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 1227.75 | 1230.22 | 1213.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 15:15:00 | 1220.35 | 1227.27 | 1218.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1220.35 | 1227.27 | 1218.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 1220.95 | 1227.27 | 1218.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1224.20 | 1226.65 | 1218.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 1217.05 | 1226.65 | 1218.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1231.75 | 1229.30 | 1223.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 1225.25 | 1229.30 | 1223.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 1351.10 | 1369.15 | 1354.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 1351.10 | 1369.15 | 1354.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 1364.00 | 1368.12 | 1355.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 1343.90 | 1368.12 | 1355.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1338.50 | 1362.19 | 1353.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 1340.80 | 1362.19 | 1353.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1339.85 | 1357.72 | 1352.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:15:00 | 1342.95 | 1354.76 | 1351.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 14:15:00 | 1329.70 | 1346.05 | 1347.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 1329.70 | 1346.05 | 1347.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 10:15:00 | 1320.75 | 1336.27 | 1342.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 14:15:00 | 1333.40 | 1325.82 | 1334.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 14:15:00 | 1333.40 | 1325.82 | 1334.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1333.40 | 1325.82 | 1334.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 1333.40 | 1325.82 | 1334.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1326.70 | 1325.37 | 1332.89 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 15:15:00 | 1342.00 | 1329.78 | 1329.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 09:15:00 | 1356.20 | 1335.07 | 1331.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 11:15:00 | 1329.75 | 1335.58 | 1332.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 1329.75 | 1335.58 | 1332.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1329.75 | 1335.58 | 1332.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 1329.75 | 1335.58 | 1332.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1317.20 | 1331.90 | 1331.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 1319.10 | 1331.90 | 1331.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 1319.65 | 1329.45 | 1330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 14:15:00 | 1312.70 | 1326.10 | 1328.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1310.10 | 1300.15 | 1310.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1310.10 | 1300.15 | 1310.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1310.10 | 1300.15 | 1310.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1314.70 | 1300.15 | 1310.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1332.85 | 1306.69 | 1312.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1332.85 | 1306.69 | 1312.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 1332.80 | 1311.91 | 1314.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:30:00 | 1329.15 | 1315.03 | 1315.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 1333.00 | 1318.62 | 1317.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 1333.00 | 1318.62 | 1317.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 1334.05 | 1324.88 | 1320.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 1315.20 | 1325.16 | 1322.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 1315.20 | 1325.16 | 1322.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1315.20 | 1325.16 | 1322.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 1315.20 | 1325.16 | 1322.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1321.00 | 1324.33 | 1322.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1333.20 | 1324.33 | 1322.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 1327.00 | 1345.17 | 1345.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 1327.00 | 1345.17 | 1345.22 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 14:15:00 | 1367.00 | 1342.92 | 1341.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 15:15:00 | 1370.00 | 1348.33 | 1343.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 1344.65 | 1352.96 | 1347.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 11:15:00 | 1344.65 | 1352.96 | 1347.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 1344.65 | 1352.96 | 1347.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 1344.65 | 1352.96 | 1347.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1354.75 | 1353.32 | 1348.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 1343.65 | 1353.32 | 1348.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1341.00 | 1353.89 | 1350.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:45:00 | 1345.85 | 1351.66 | 1349.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 12:15:00 | 1336.75 | 1346.22 | 1347.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 1336.75 | 1346.22 | 1347.50 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 15:15:00 | 1356.05 | 1348.10 | 1348.01 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1320.80 | 1342.64 | 1345.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 1261.50 | 1297.07 | 1319.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 1228.65 | 1222.33 | 1248.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:00:00 | 1228.65 | 1222.33 | 1248.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1248.35 | 1231.61 | 1246.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:45:00 | 1221.55 | 1240.34 | 1243.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 12:15:00 | 1248.35 | 1242.11 | 1241.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 12:15:00 | 1248.35 | 1242.11 | 1241.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 13:15:00 | 1261.40 | 1245.97 | 1243.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 15:15:00 | 1243.90 | 1246.52 | 1244.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 15:15:00 | 1243.90 | 1246.52 | 1244.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1243.90 | 1246.52 | 1244.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1223.30 | 1246.52 | 1244.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 1220.05 | 1241.23 | 1242.19 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 1237.40 | 1231.23 | 1230.65 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 1211.75 | 1227.97 | 1229.41 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 14:15:00 | 1239.00 | 1225.86 | 1225.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 12:15:00 | 1240.95 | 1232.01 | 1228.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 1284.05 | 1285.24 | 1266.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 10:00:00 | 1284.05 | 1285.24 | 1266.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1319.00 | 1328.84 | 1319.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1319.00 | 1328.84 | 1319.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1326.50 | 1328.37 | 1320.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 1349.30 | 1326.88 | 1323.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 1370.00 | 1392.95 | 1395.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 1370.00 | 1392.95 | 1395.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 09:15:00 | 1354.40 | 1377.52 | 1382.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 1367.50 | 1360.62 | 1369.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1367.50 | 1360.62 | 1369.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1367.50 | 1360.62 | 1369.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 1367.10 | 1360.62 | 1369.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1369.90 | 1362.47 | 1369.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 1369.90 | 1362.47 | 1369.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1364.15 | 1362.81 | 1368.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 1370.40 | 1362.81 | 1368.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 1365.55 | 1363.36 | 1368.64 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 1386.90 | 1374.07 | 1372.34 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 1370.00 | 1373.78 | 1373.82 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 1381.35 | 1375.29 | 1374.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 14:15:00 | 1401.55 | 1380.54 | 1376.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 10:15:00 | 1378.40 | 1382.46 | 1379.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 10:15:00 | 1378.40 | 1382.46 | 1379.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1378.40 | 1382.46 | 1379.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 1378.40 | 1382.46 | 1379.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 1379.30 | 1381.83 | 1379.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:45:00 | 1379.50 | 1381.83 | 1379.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 1378.35 | 1381.13 | 1378.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:00:00 | 1378.35 | 1381.13 | 1378.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 1407.90 | 1386.49 | 1381.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:30:00 | 1378.75 | 1386.49 | 1381.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1384.00 | 1388.29 | 1384.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:00:00 | 1384.00 | 1388.29 | 1384.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1383.80 | 1387.39 | 1384.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:45:00 | 1384.60 | 1387.39 | 1384.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1379.05 | 1385.73 | 1384.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:00:00 | 1379.05 | 1385.73 | 1384.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1380.70 | 1384.72 | 1383.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:30:00 | 1381.10 | 1384.72 | 1383.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1375.00 | 1382.78 | 1383.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 1349.85 | 1375.87 | 1379.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 13:15:00 | 1374.80 | 1370.50 | 1375.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 13:15:00 | 1374.80 | 1370.50 | 1375.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1374.80 | 1370.50 | 1375.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:45:00 | 1379.70 | 1370.50 | 1375.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1374.50 | 1371.30 | 1375.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 1360.05 | 1372.01 | 1375.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:15:00 | 1292.05 | 1310.04 | 1325.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 12:15:00 | 1323.20 | 1307.50 | 1320.21 | SL hit (close>ema200) qty=0.50 sl=1307.50 alert=retest2 |

### Cycle 111 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 1341.95 | 1324.92 | 1324.79 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 1315.25 | 1323.32 | 1324.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 10:15:00 | 1312.00 | 1318.48 | 1321.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 11:15:00 | 1333.05 | 1321.39 | 1322.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 11:15:00 | 1333.05 | 1321.39 | 1322.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1333.05 | 1321.39 | 1322.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 1333.05 | 1321.39 | 1322.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 12:15:00 | 1330.70 | 1323.25 | 1323.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 13:15:00 | 1336.95 | 1325.99 | 1324.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 12:15:00 | 1360.55 | 1363.64 | 1349.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:45:00 | 1361.80 | 1363.64 | 1349.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 1362.10 | 1372.46 | 1361.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 1362.10 | 1372.46 | 1361.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 1361.50 | 1370.26 | 1361.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:45:00 | 1364.05 | 1370.26 | 1361.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1350.00 | 1366.21 | 1360.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:45:00 | 1350.50 | 1366.21 | 1360.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1349.00 | 1362.77 | 1359.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 1326.05 | 1362.77 | 1359.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 1328.00 | 1355.82 | 1356.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 1313.30 | 1333.73 | 1344.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1327.95 | 1323.72 | 1334.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:45:00 | 1330.00 | 1323.72 | 1334.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1330.40 | 1325.05 | 1334.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:30:00 | 1330.05 | 1325.05 | 1334.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1332.10 | 1326.46 | 1334.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:45:00 | 1332.25 | 1326.46 | 1334.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1342.10 | 1329.59 | 1334.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 1342.10 | 1329.59 | 1334.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1360.00 | 1335.67 | 1337.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1332.00 | 1335.67 | 1337.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1351.00 | 1327.00 | 1324.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1351.00 | 1327.00 | 1324.92 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 1312.80 | 1324.00 | 1325.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 1302.80 | 1315.84 | 1319.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 1319.70 | 1309.71 | 1314.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 1319.70 | 1309.71 | 1314.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1319.70 | 1309.71 | 1314.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:45:00 | 1321.10 | 1309.71 | 1314.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1301.40 | 1308.05 | 1313.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:30:00 | 1298.50 | 1306.39 | 1311.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 1299.35 | 1306.39 | 1311.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:15:00 | 1298.80 | 1305.08 | 1310.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 1296.45 | 1303.42 | 1309.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1288.00 | 1285.24 | 1292.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:30:00 | 1299.40 | 1285.24 | 1292.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1298.45 | 1287.88 | 1293.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 1298.45 | 1287.88 | 1293.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1299.00 | 1290.11 | 1293.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 1318.50 | 1290.11 | 1293.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1299.70 | 1291.09 | 1293.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:30:00 | 1300.00 | 1291.09 | 1293.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1296.10 | 1292.09 | 1293.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1290.40 | 1294.22 | 1294.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 1233.58 | 1257.86 | 1270.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 1234.38 | 1257.86 | 1270.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 1233.86 | 1257.86 | 1270.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 1231.63 | 1257.86 | 1270.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1225.88 | 1253.66 | 1266.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-28 09:15:00 | 1168.65 | 1186.87 | 1208.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 1231.05 | 1205.46 | 1203.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 15:15:00 | 1259.00 | 1224.65 | 1213.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1263.35 | 1277.17 | 1261.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1263.35 | 1277.17 | 1261.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1263.35 | 1277.17 | 1261.85 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 1239.00 | 1259.70 | 1260.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 12:15:00 | 1232.10 | 1250.75 | 1255.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 1262.65 | 1240.06 | 1245.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 12:15:00 | 1262.65 | 1240.06 | 1245.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 1262.65 | 1240.06 | 1245.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:45:00 | 1259.25 | 1240.06 | 1245.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 1267.15 | 1245.48 | 1247.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:45:00 | 1274.65 | 1245.48 | 1247.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 1278.20 | 1252.02 | 1250.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 11:15:00 | 1295.00 | 1270.47 | 1260.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 1267.80 | 1273.33 | 1264.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:45:00 | 1273.95 | 1273.33 | 1264.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1266.35 | 1272.20 | 1265.71 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1248.00 | 1261.76 | 1262.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1243.75 | 1258.16 | 1260.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1255.20 | 1254.32 | 1257.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 1255.20 | 1254.32 | 1257.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1255.20 | 1254.32 | 1257.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 1255.20 | 1254.32 | 1257.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1258.80 | 1255.22 | 1258.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:45:00 | 1262.00 | 1255.22 | 1258.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1254.20 | 1255.01 | 1257.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:30:00 | 1259.70 | 1255.01 | 1257.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1254.05 | 1252.41 | 1255.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1257.40 | 1252.41 | 1255.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1258.90 | 1253.71 | 1255.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1258.90 | 1253.71 | 1255.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1251.45 | 1253.26 | 1255.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:15:00 | 1246.85 | 1252.73 | 1254.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 12:15:00 | 1184.51 | 1219.57 | 1235.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 1207.90 | 1202.94 | 1217.07 | SL hit (close>ema200) qty=0.50 sl=1202.94 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 1145.00 | 1138.25 | 1137.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 15:15:00 | 1162.75 | 1146.62 | 1142.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1142.60 | 1147.97 | 1144.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 12:15:00 | 1142.60 | 1147.97 | 1144.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1142.60 | 1147.97 | 1144.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 1142.60 | 1147.97 | 1144.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1140.55 | 1146.49 | 1144.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1140.55 | 1146.49 | 1144.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1143.65 | 1145.92 | 1144.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1143.65 | 1145.92 | 1144.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1142.95 | 1145.33 | 1143.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 1145.35 | 1145.33 | 1143.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1138.95 | 1144.05 | 1143.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 1138.95 | 1144.05 | 1143.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 1133.00 | 1141.84 | 1142.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 13:15:00 | 1129.85 | 1136.21 | 1138.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1144.20 | 1136.70 | 1138.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1144.20 | 1136.70 | 1138.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1144.20 | 1136.70 | 1138.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 1150.15 | 1136.70 | 1138.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1147.50 | 1138.86 | 1139.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 1147.50 | 1138.86 | 1139.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 1150.40 | 1141.17 | 1140.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1168.20 | 1152.69 | 1147.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 1275.80 | 1276.61 | 1253.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 10:00:00 | 1275.80 | 1276.61 | 1253.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1281.75 | 1289.57 | 1281.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:45:00 | 1282.80 | 1289.57 | 1281.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1288.80 | 1289.42 | 1282.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1303.80 | 1285.28 | 1282.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 1273.35 | 1282.89 | 1281.34 | SL hit (close<static) qty=1.00 sl=1280.05 alert=retest2 |

### Cycle 124 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 1310.90 | 1316.50 | 1316.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1274.35 | 1308.07 | 1312.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 1300.00 | 1299.83 | 1306.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 1300.00 | 1299.83 | 1306.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1292.05 | 1298.30 | 1304.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:15:00 | 1288.05 | 1298.77 | 1303.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 1288.75 | 1276.44 | 1276.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 1288.75 | 1276.44 | 1276.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1312.65 | 1294.69 | 1286.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 1301.10 | 1301.71 | 1292.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 10:00:00 | 1301.10 | 1301.71 | 1292.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 1291.80 | 1298.74 | 1292.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 1291.80 | 1298.74 | 1292.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 1288.95 | 1296.78 | 1292.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 1288.95 | 1296.78 | 1292.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1285.00 | 1294.42 | 1291.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 1285.00 | 1294.42 | 1291.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1275.15 | 1290.57 | 1290.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1275.15 | 1290.57 | 1290.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 1281.00 | 1288.65 | 1289.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1268.90 | 1284.70 | 1287.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 1289.25 | 1283.30 | 1285.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 1289.25 | 1283.30 | 1285.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1289.25 | 1283.30 | 1285.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 1289.25 | 1283.30 | 1285.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1298.15 | 1286.27 | 1286.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 1297.90 | 1286.27 | 1286.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 1300.00 | 1289.01 | 1288.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 1314.95 | 1297.85 | 1292.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1301.20 | 1306.38 | 1299.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 1301.20 | 1306.38 | 1299.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1301.20 | 1306.38 | 1299.77 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 13:15:00 | 1287.65 | 1294.83 | 1295.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 14:15:00 | 1274.15 | 1290.69 | 1293.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 1305.55 | 1292.11 | 1293.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 1305.55 | 1292.11 | 1293.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1305.55 | 1292.11 | 1293.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:45:00 | 1303.70 | 1292.11 | 1293.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1296.25 | 1292.94 | 1293.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:30:00 | 1284.65 | 1290.08 | 1292.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1220.42 | 1244.28 | 1248.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 1156.19 | 1184.01 | 1208.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 129 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 1209.25 | 1185.99 | 1183.04 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1156.60 | 1180.30 | 1183.34 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 1187.80 | 1182.56 | 1182.38 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1162.25 | 1178.50 | 1180.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1154.65 | 1173.73 | 1178.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1132.00 | 1125.45 | 1141.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1132.00 | 1125.45 | 1141.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1088.90 | 1108.83 | 1124.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:45:00 | 1085.45 | 1105.06 | 1121.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 1085.55 | 1105.06 | 1121.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:45:00 | 1084.40 | 1098.98 | 1115.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1031.18 | 1069.15 | 1094.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1031.27 | 1069.15 | 1094.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1030.18 | 1069.15 | 1094.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 10:15:00 | 976.91 | 1019.55 | 1052.12 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 1056.95 | 1040.60 | 1038.95 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 1022.65 | 1038.79 | 1039.08 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 1053.00 | 1040.70 | 1039.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1059.70 | 1044.50 | 1041.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1034.05 | 1055.46 | 1050.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1034.05 | 1055.46 | 1050.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1034.05 | 1055.46 | 1050.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1042.50 | 1055.46 | 1050.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1030.20 | 1050.41 | 1048.60 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 1026.70 | 1045.67 | 1046.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1001.95 | 1035.20 | 1041.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1017.50 | 1007.37 | 1020.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1017.50 | 1007.37 | 1020.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1017.50 | 1007.37 | 1020.53 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 1044.80 | 1025.70 | 1023.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 1053.95 | 1037.37 | 1030.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 15:15:00 | 1045.05 | 1046.61 | 1038.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:15:00 | 1034.10 | 1046.61 | 1038.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1026.70 | 1042.63 | 1037.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 1026.70 | 1042.63 | 1037.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1043.20 | 1042.74 | 1038.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1046.60 | 1042.74 | 1038.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 1023.65 | 1036.94 | 1037.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1023.65 | 1036.94 | 1037.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1012.10 | 1028.65 | 1032.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 990.00 | 975.49 | 989.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 990.00 | 975.49 | 989.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 990.00 | 975.49 | 989.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 980.10 | 975.49 | 989.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 980.00 | 976.39 | 988.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 975.30 | 980.59 | 987.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 926.53 | 968.31 | 980.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 12:15:00 | 877.77 | 930.83 | 958.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 139 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 936.00 | 921.67 | 921.13 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 922.00 | 925.90 | 925.94 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 11:15:00 | 932.15 | 926.63 | 926.05 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 911.20 | 923.40 | 924.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 891.35 | 906.83 | 914.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 900.45 | 899.91 | 907.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 14:30:00 | 902.25 | 899.91 | 907.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 881.25 | 876.43 | 889.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 881.25 | 876.43 | 889.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 834.15 | 867.16 | 883.04 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 887.85 | 862.80 | 861.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 912.85 | 877.36 | 868.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 925.30 | 930.02 | 912.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 925.30 | 930.02 | 912.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 907.00 | 917.45 | 913.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 907.00 | 917.45 | 913.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 910.00 | 915.96 | 913.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 887.90 | 915.96 | 913.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 891.85 | 911.14 | 911.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 14:15:00 | 873.35 | 891.96 | 900.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 893.00 | 890.54 | 897.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 893.00 | 890.54 | 897.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 893.00 | 890.54 | 897.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 895.95 | 890.54 | 897.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 871.20 | 868.65 | 875.48 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 886.85 | 878.92 | 877.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 904.45 | 886.20 | 881.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 10:15:00 | 1005.70 | 1007.05 | 989.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 11:00:00 | 1005.70 | 1007.05 | 989.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1001.40 | 1004.14 | 995.21 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 981.00 | 993.18 | 993.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 09:15:00 | 964.00 | 972.86 | 977.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 13:15:00 | 882.40 | 879.74 | 899.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 14:00:00 | 882.40 | 879.74 | 899.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 889.15 | 877.13 | 886.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:00:00 | 889.15 | 877.13 | 886.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 881.95 | 878.10 | 885.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 900.40 | 878.10 | 885.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 903.35 | 883.15 | 887.54 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 896.45 | 890.98 | 890.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 951.50 | 905.98 | 897.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 931.00 | 931.22 | 918.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 925.55 | 929.86 | 923.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 925.55 | 929.86 | 923.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 938.35 | 932.61 | 925.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 941.15 | 937.83 | 931.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 15:15:00 | 1032.19 | 982.24 | 956.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 949.60 | 979.04 | 981.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 944.50 | 959.28 | 969.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 957.95 | 951.02 | 959.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 15:15:00 | 957.95 | 951.02 | 959.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 957.95 | 951.02 | 959.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 961.15 | 951.02 | 959.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 958.15 | 952.45 | 959.35 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 975.00 | 961.87 | 961.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 14:15:00 | 976.20 | 968.39 | 965.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 955.55 | 965.82 | 964.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 15:15:00 | 955.55 | 965.82 | 964.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 955.55 | 965.82 | 964.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 960.75 | 965.82 | 964.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 975.75 | 967.81 | 965.69 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 09:15:00 | 963.15 | 965.13 | 965.38 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 971.20 | 966.25 | 965.84 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 15:15:00 | 961.05 | 965.52 | 965.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 949.45 | 962.31 | 964.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 946.70 | 936.21 | 946.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 946.70 | 936.21 | 946.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 946.70 | 936.21 | 946.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 946.95 | 936.21 | 946.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 942.95 | 937.56 | 945.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:30:00 | 939.60 | 938.53 | 945.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 973.65 | 945.55 | 948.08 | SL hit (close>static) qty=1.00 sl=953.65 alert=retest2 |

### Cycle 153 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 968.15 | 950.07 | 949.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 985.05 | 959.24 | 954.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 963.10 | 966.02 | 959.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 13:45:00 | 968.15 | 966.02 | 959.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 948.45 | 962.50 | 958.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 948.45 | 962.50 | 958.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 943.00 | 958.60 | 956.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 921.15 | 958.60 | 956.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 929.00 | 952.68 | 954.27 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 986.70 | 951.46 | 949.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1001.75 | 977.11 | 964.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 1109.15 | 1118.26 | 1102.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 15:00:00 | 1109.15 | 1118.26 | 1102.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1102.15 | 1113.39 | 1102.63 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 1076.20 | 1095.22 | 1096.58 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 1105.50 | 1098.10 | 1097.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1118.90 | 1105.86 | 1102.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1107.45 | 1113.26 | 1108.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1107.45 | 1113.26 | 1108.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1107.45 | 1113.26 | 1108.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 1132.15 | 1116.99 | 1111.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:30:00 | 1128.35 | 1117.00 | 1114.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1127.80 | 1118.40 | 1115.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 1129.10 | 1120.26 | 1116.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1114.00 | 1119.57 | 1116.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1114.00 | 1119.57 | 1116.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1113.50 | 1118.35 | 1116.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1115.80 | 1117.23 | 1116.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 1121.50 | 1116.79 | 1116.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1123.60 | 1118.15 | 1116.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1138.70 | 1148.95 | 1140.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 1138.70 | 1148.95 | 1140.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1138.70 | 1148.95 | 1140.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1138.70 | 1148.95 | 1140.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1171.60 | 1153.48 | 1143.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1179.50 | 1153.48 | 1143.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:30:00 | 1173.90 | 1163.97 | 1155.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 1151.00 | 1154.91 | 1155.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 1151.00 | 1154.91 | 1155.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 1143.60 | 1152.65 | 1154.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 11:15:00 | 1152.00 | 1151.81 | 1153.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 11:15:00 | 1152.00 | 1151.81 | 1153.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1152.00 | 1151.81 | 1153.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 1153.10 | 1151.81 | 1153.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1171.90 | 1155.82 | 1155.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1189.40 | 1162.52 | 1158.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 1183.60 | 1185.52 | 1174.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 1181.80 | 1185.52 | 1174.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1172.10 | 1181.52 | 1174.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 1171.80 | 1181.52 | 1174.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1171.30 | 1179.47 | 1174.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 1172.80 | 1177.60 | 1173.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 1167.10 | 1174.12 | 1172.75 | SL hit (close<static) qty=1.00 sl=1168.60 alert=retest2 |

### Cycle 162 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1161.30 | 1174.63 | 1174.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1144.20 | 1158.98 | 1164.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1163.50 | 1157.43 | 1162.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 1163.50 | 1157.43 | 1162.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1163.50 | 1157.43 | 1162.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1163.50 | 1157.43 | 1162.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1176.50 | 1161.24 | 1163.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1176.50 | 1161.24 | 1163.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1176.90 | 1164.37 | 1164.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1178.00 | 1164.37 | 1164.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1173.20 | 1166.14 | 1165.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1200.50 | 1173.01 | 1168.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 1184.90 | 1191.13 | 1183.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:45:00 | 1184.10 | 1191.13 | 1183.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1195.70 | 1192.04 | 1184.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 1197.10 | 1192.04 | 1184.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1199.80 | 1198.66 | 1191.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1192.10 | 1198.66 | 1191.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1192.90 | 1197.51 | 1191.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1192.90 | 1197.51 | 1191.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1189.70 | 1195.95 | 1191.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 1184.40 | 1195.95 | 1191.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1192.20 | 1195.20 | 1191.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 1185.10 | 1195.20 | 1191.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1192.40 | 1194.64 | 1191.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:15:00 | 1190.00 | 1194.64 | 1191.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1190.00 | 1193.71 | 1191.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 1176.20 | 1193.71 | 1191.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 1168.40 | 1188.65 | 1189.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1161.20 | 1177.32 | 1183.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1179.50 | 1171.29 | 1177.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1179.50 | 1171.29 | 1177.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1179.50 | 1171.29 | 1177.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1179.50 | 1171.29 | 1177.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1184.30 | 1173.89 | 1178.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 1184.30 | 1173.89 | 1178.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1186.10 | 1176.34 | 1179.23 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1188.60 | 1182.02 | 1181.39 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 1180.00 | 1181.21 | 1181.31 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 15:15:00 | 1182.00 | 1181.46 | 1181.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1214.30 | 1188.03 | 1184.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 1225.70 | 1227.82 | 1216.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1225.70 | 1227.82 | 1216.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1219.10 | 1225.48 | 1217.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 1240.80 | 1225.97 | 1222.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 1211.80 | 1221.04 | 1220.85 | SL hit (close<static) qty=1.00 sl=1212.10 alert=retest2 |

### Cycle 168 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1210.10 | 1218.85 | 1219.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1193.10 | 1208.16 | 1213.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 1199.00 | 1198.03 | 1205.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 1199.00 | 1198.03 | 1205.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1208.70 | 1198.64 | 1204.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1208.70 | 1198.64 | 1204.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1203.90 | 1199.69 | 1204.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 1209.20 | 1199.69 | 1204.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1203.90 | 1200.53 | 1204.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 1203.90 | 1200.53 | 1204.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1204.60 | 1201.35 | 1204.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 1204.50 | 1201.35 | 1204.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1203.30 | 1201.74 | 1204.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 1198.50 | 1200.93 | 1203.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 1198.90 | 1198.55 | 1202.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1196.00 | 1188.69 | 1188.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1196.00 | 1188.69 | 1188.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 1208.30 | 1194.88 | 1191.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1197.50 | 1198.81 | 1195.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:45:00 | 1198.80 | 1198.81 | 1195.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1196.00 | 1198.25 | 1195.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1196.10 | 1198.25 | 1195.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1202.70 | 1199.14 | 1195.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:15:00 | 1197.80 | 1199.14 | 1195.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1199.60 | 1199.23 | 1196.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 1196.00 | 1199.23 | 1196.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1197.90 | 1198.96 | 1196.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1197.90 | 1198.96 | 1196.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1201.00 | 1199.54 | 1197.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:15:00 | 1186.90 | 1199.54 | 1197.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1185.80 | 1196.79 | 1196.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1185.80 | 1196.79 | 1196.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1193.90 | 1196.21 | 1196.07 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1194.10 | 1195.79 | 1195.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 1191.10 | 1194.26 | 1195.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1199.40 | 1194.61 | 1195.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1199.40 | 1194.61 | 1195.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1199.40 | 1194.61 | 1195.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1199.40 | 1194.61 | 1195.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1192.10 | 1194.11 | 1194.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 1188.80 | 1194.11 | 1194.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 1190.20 | 1192.65 | 1194.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 1189.30 | 1192.68 | 1193.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 1197.00 | 1194.72 | 1194.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1197.00 | 1194.72 | 1194.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 1202.10 | 1196.66 | 1195.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1203.10 | 1205.03 | 1200.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:30:00 | 1205.00 | 1205.03 | 1200.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1197.60 | 1203.54 | 1200.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1197.60 | 1203.54 | 1200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1197.90 | 1202.41 | 1200.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1195.40 | 1202.41 | 1200.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1192.10 | 1200.35 | 1199.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 1192.10 | 1200.35 | 1199.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1198.80 | 1199.06 | 1198.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1224.00 | 1199.06 | 1198.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 1185.00 | 1196.43 | 1197.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1185.00 | 1196.43 | 1197.85 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1205.70 | 1195.67 | 1195.11 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 1189.90 | 1194.77 | 1194.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 1184.20 | 1190.92 | 1192.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1195.30 | 1190.70 | 1192.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1195.30 | 1190.70 | 1192.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1195.30 | 1190.70 | 1192.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 1195.30 | 1190.70 | 1192.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1193.80 | 1191.32 | 1192.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 1188.60 | 1191.30 | 1192.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 1189.00 | 1188.41 | 1190.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 1199.40 | 1192.10 | 1191.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 1199.40 | 1192.10 | 1191.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 1204.00 | 1194.75 | 1193.17 | Break + close above crossover candle high |

### Cycle 176 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1181.00 | 1192.00 | 1192.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1165.00 | 1186.60 | 1189.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1132.50 | 1127.29 | 1134.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1132.50 | 1127.29 | 1134.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1132.50 | 1127.29 | 1134.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 1125.10 | 1127.29 | 1134.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1135.50 | 1128.94 | 1134.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 1135.50 | 1128.94 | 1134.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1140.00 | 1131.15 | 1134.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 1141.60 | 1131.15 | 1134.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1146.10 | 1134.14 | 1135.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 1145.30 | 1134.14 | 1135.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 14:15:00 | 1145.10 | 1138.16 | 1137.59 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1131.90 | 1136.67 | 1137.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1125.70 | 1134.00 | 1135.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1122.20 | 1121.98 | 1127.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 1122.20 | 1121.98 | 1127.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1122.20 | 1121.98 | 1127.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 1123.70 | 1121.98 | 1127.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1125.00 | 1122.58 | 1127.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1130.00 | 1122.58 | 1127.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1121.80 | 1122.43 | 1126.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 1117.00 | 1125.86 | 1127.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 1116.60 | 1123.40 | 1125.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 1115.90 | 1122.36 | 1125.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 1116.10 | 1119.07 | 1122.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1116.20 | 1117.20 | 1121.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1129.80 | 1117.20 | 1121.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1118.40 | 1117.44 | 1121.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:15:00 | 1111.30 | 1117.44 | 1121.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1233.80 | 1182.68 | 1158.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 1220.10 | 1223.65 | 1199.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:30:00 | 1223.10 | 1223.65 | 1199.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1222.80 | 1228.31 | 1220.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 1222.80 | 1228.31 | 1220.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1227.00 | 1227.39 | 1221.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 1227.20 | 1227.39 | 1221.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1231.70 | 1228.26 | 1222.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 1230.10 | 1228.26 | 1222.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1271.50 | 1272.75 | 1261.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:45:00 | 1276.90 | 1273.58 | 1262.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 1276.10 | 1274.44 | 1264.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:30:00 | 1277.80 | 1275.48 | 1266.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 1292.80 | 1275.77 | 1268.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1298.90 | 1292.02 | 1282.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1287.40 | 1292.02 | 1282.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1292.80 | 1297.81 | 1291.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1292.80 | 1297.81 | 1291.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1292.60 | 1296.77 | 1291.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 1292.30 | 1296.77 | 1291.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1292.60 | 1295.93 | 1291.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:15:00 | 1288.00 | 1295.93 | 1291.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1289.00 | 1294.55 | 1291.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1289.00 | 1294.55 | 1291.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1288.00 | 1293.24 | 1291.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 1284.30 | 1293.24 | 1291.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1258.70 | 1280.18 | 1285.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1280.60 | 1270.86 | 1277.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 1280.60 | 1270.86 | 1277.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1280.60 | 1270.86 | 1277.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 1280.60 | 1270.86 | 1277.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1285.30 | 1273.75 | 1277.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 1285.70 | 1273.75 | 1277.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1276.50 | 1275.43 | 1277.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:45:00 | 1277.70 | 1275.43 | 1277.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1270.00 | 1274.34 | 1276.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 1264.10 | 1274.34 | 1276.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 1263.20 | 1270.01 | 1274.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 1263.50 | 1269.13 | 1273.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 1265.00 | 1268.44 | 1272.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1247.30 | 1257.44 | 1265.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:30:00 | 1244.90 | 1253.45 | 1262.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:00:00 | 1241.90 | 1243.00 | 1252.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 1245.90 | 1235.80 | 1243.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:45:00 | 1240.00 | 1237.58 | 1243.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1250.50 | 1240.16 | 1243.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:00:00 | 1250.50 | 1240.16 | 1243.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1257.00 | 1243.53 | 1245.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1257.00 | 1243.53 | 1245.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1274.30 | 1253.30 | 1249.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1260.10 | 1261.73 | 1256.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1260.10 | 1261.73 | 1256.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1267.30 | 1262.53 | 1258.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 1270.30 | 1264.64 | 1259.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1276.30 | 1268.02 | 1262.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1273.10 | 1267.74 | 1265.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:45:00 | 1272.40 | 1271.88 | 1268.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1271.30 | 1272.15 | 1269.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 1287.00 | 1273.46 | 1270.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 1276.80 | 1274.54 | 1271.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1295.10 | 1273.30 | 1270.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 13:15:00 | 1302.20 | 1292.28 | 1283.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1290.10 | 1291.85 | 1284.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1290.10 | 1291.85 | 1284.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1291.00 | 1292.34 | 1286.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 1300.80 | 1293.28 | 1287.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:00:00 | 1303.00 | 1296.25 | 1289.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 1301.10 | 1300.04 | 1294.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1282.70 | 1296.57 | 1293.52 | SL hit (close<static) qty=1.00 sl=1285.50 alert=retest2 |

### Cycle 184 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1281.10 | 1291.45 | 1291.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1278.90 | 1287.72 | 1289.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 1266.20 | 1265.85 | 1270.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 1266.20 | 1265.85 | 1270.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1266.20 | 1265.85 | 1270.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1266.20 | 1265.85 | 1270.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1260.00 | 1264.68 | 1269.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1258.60 | 1264.68 | 1269.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1266.70 | 1265.09 | 1269.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:00:00 | 1254.60 | 1261.36 | 1266.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 1252.80 | 1257.77 | 1264.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 1252.70 | 1251.13 | 1256.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:30:00 | 1253.60 | 1256.06 | 1256.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1270.50 | 1260.16 | 1258.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 1253.90 | 1262.35 | 1260.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 1253.90 | 1262.35 | 1260.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1253.90 | 1262.35 | 1260.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:30:00 | 1257.10 | 1262.35 | 1260.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1246.00 | 1259.08 | 1259.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1245.10 | 1256.28 | 1258.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1255.90 | 1245.61 | 1250.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1255.90 | 1245.61 | 1250.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1255.90 | 1245.61 | 1250.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1253.20 | 1245.61 | 1250.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1247.60 | 1246.01 | 1250.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 1244.60 | 1246.01 | 1250.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1245.50 | 1241.16 | 1245.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 1244.70 | 1241.97 | 1245.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 1245.60 | 1243.04 | 1245.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1247.00 | 1243.83 | 1245.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 1247.00 | 1243.83 | 1245.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1254.50 | 1245.97 | 1246.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1254.50 | 1245.97 | 1246.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 11:15:00 | 1239.50 | 1245.26 | 1245.96 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1249.70 | 1246.36 | 1246.35 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 1241.80 | 1245.45 | 1245.94 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1253.00 | 1246.32 | 1245.62 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 1238.10 | 1244.67 | 1244.94 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 1247.90 | 1245.44 | 1245.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 1253.80 | 1247.04 | 1246.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1257.20 | 1262.35 | 1256.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1257.20 | 1262.35 | 1256.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1257.20 | 1262.35 | 1256.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 1257.90 | 1262.35 | 1256.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1253.80 | 1260.64 | 1255.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1253.80 | 1260.64 | 1255.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1258.10 | 1260.13 | 1256.17 | EMA400 retest candle locked (from upside) |

### Cycle 194 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1243.80 | 1253.42 | 1253.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 1242.20 | 1251.18 | 1252.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 1254.10 | 1249.93 | 1251.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 1254.10 | 1249.93 | 1251.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1254.10 | 1249.93 | 1251.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1254.10 | 1249.93 | 1251.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1257.30 | 1251.40 | 1252.14 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 1258.00 | 1253.57 | 1253.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1259.60 | 1254.77 | 1253.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1252.00 | 1254.51 | 1253.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 13:15:00 | 1252.00 | 1254.51 | 1253.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 1252.00 | 1254.51 | 1253.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 1252.00 | 1254.51 | 1253.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1249.60 | 1253.53 | 1253.52 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 1250.60 | 1252.94 | 1253.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 1242.00 | 1250.76 | 1252.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 1253.30 | 1248.70 | 1250.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1253.30 | 1248.70 | 1250.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1253.30 | 1248.70 | 1250.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 1251.80 | 1248.70 | 1250.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1261.40 | 1251.24 | 1251.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 1260.80 | 1251.24 | 1251.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 1260.00 | 1252.99 | 1252.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 1276.80 | 1257.76 | 1254.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 1265.00 | 1265.09 | 1260.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:15:00 | 1264.50 | 1265.09 | 1260.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1262.50 | 1264.57 | 1260.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 1259.00 | 1264.57 | 1260.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1263.30 | 1264.32 | 1260.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1262.50 | 1264.32 | 1260.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1253.00 | 1262.05 | 1259.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 1253.00 | 1262.05 | 1259.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1247.20 | 1259.08 | 1258.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 1248.50 | 1259.08 | 1258.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 1250.60 | 1257.39 | 1258.06 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1277.50 | 1259.69 | 1258.77 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1253.70 | 1264.52 | 1264.74 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 1271.40 | 1265.90 | 1265.35 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1249.90 | 1262.09 | 1263.67 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1279.90 | 1265.32 | 1264.68 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 1260.10 | 1263.97 | 1264.23 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1276.00 | 1266.35 | 1265.26 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1257.20 | 1264.58 | 1265.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 1250.20 | 1261.47 | 1263.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1245.90 | 1238.24 | 1245.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1245.90 | 1238.24 | 1245.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1245.90 | 1238.24 | 1245.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1264.80 | 1238.24 | 1245.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1246.50 | 1239.89 | 1245.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 1247.00 | 1239.89 | 1245.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1249.90 | 1241.89 | 1246.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 1252.90 | 1241.89 | 1246.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1267.50 | 1247.01 | 1248.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 1267.50 | 1247.01 | 1248.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 1261.40 | 1249.89 | 1249.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1274.00 | 1257.91 | 1253.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 12:15:00 | 1258.50 | 1258.69 | 1255.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:45:00 | 1259.10 | 1258.69 | 1255.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1259.40 | 1258.83 | 1255.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 1264.90 | 1257.85 | 1255.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 15:15:00 | 1265.00 | 1257.85 | 1255.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 1262.40 | 1259.42 | 1256.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:00:00 | 1261.10 | 1260.41 | 1257.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1263.00 | 1264.29 | 1261.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 1263.00 | 1264.29 | 1261.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1260.90 | 1263.61 | 1261.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 1262.20 | 1263.61 | 1261.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | SL hit (close<static) qty=1.00 sl=1254.30 alert=retest2 |

### Cycle 208 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 1284.60 | 1294.65 | 1294.84 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1298.40 | 1295.40 | 1295.16 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 1287.10 | 1293.74 | 1294.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 1280.90 | 1291.17 | 1293.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 1262.00 | 1260.72 | 1270.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:45:00 | 1262.50 | 1260.72 | 1270.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1261.40 | 1261.89 | 1268.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:15:00 | 1260.10 | 1261.89 | 1268.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:45:00 | 1260.20 | 1260.89 | 1266.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 1255.40 | 1252.48 | 1255.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1197.09 | 1218.35 | 1223.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1197.19 | 1218.35 | 1223.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 1192.63 | 1208.16 | 1216.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1220.60 | 1209.52 | 1215.47 | SL hit (close>ema200) qty=0.50 sl=1209.52 alert=retest2 |

### Cycle 211 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1225.50 | 1218.67 | 1218.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1229.60 | 1222.01 | 1219.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 12:15:00 | 1223.30 | 1225.54 | 1222.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 12:15:00 | 1223.30 | 1225.54 | 1222.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1223.30 | 1225.54 | 1222.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 1223.30 | 1225.54 | 1222.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1226.50 | 1225.73 | 1222.70 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 1211.70 | 1221.12 | 1221.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 14:15:00 | 1208.00 | 1217.02 | 1219.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 1186.70 | 1186.54 | 1193.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:45:00 | 1189.10 | 1186.54 | 1193.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1190.30 | 1188.33 | 1193.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 1171.90 | 1188.33 | 1193.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1194.30 | 1190.38 | 1190.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 1194.30 | 1190.38 | 1190.27 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 1170.20 | 1186.34 | 1188.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 1163.50 | 1181.77 | 1186.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1163.10 | 1162.71 | 1171.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 15:00:00 | 1163.10 | 1162.71 | 1171.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1166.90 | 1163.54 | 1171.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 1161.70 | 1162.00 | 1169.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:45:00 | 1155.50 | 1160.14 | 1168.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1171.90 | 1163.71 | 1162.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1171.90 | 1163.71 | 1162.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1197.80 | 1171.52 | 1166.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 1173.50 | 1174.39 | 1168.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 12:00:00 | 1173.50 | 1174.39 | 1168.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1162.30 | 1171.97 | 1168.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 1162.30 | 1171.97 | 1168.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1168.50 | 1171.28 | 1168.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1173.30 | 1168.85 | 1167.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 1173.80 | 1169.38 | 1168.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 1177.90 | 1169.38 | 1168.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1173.20 | 1171.87 | 1170.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1172.40 | 1172.22 | 1170.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 1172.40 | 1172.22 | 1170.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1170.00 | 1171.78 | 1170.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 1170.00 | 1171.78 | 1170.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1171.00 | 1171.62 | 1170.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1171.30 | 1171.62 | 1170.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1175.80 | 1172.46 | 1171.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 1164.90 | 1172.46 | 1171.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1173.00 | 1172.57 | 1171.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1173.00 | 1172.57 | 1171.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 1176.60 | 1171.68 | 1171.09 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1162.70 | 1171.19 | 1172.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 1159.80 | 1168.92 | 1170.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 1167.20 | 1165.04 | 1168.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 1167.20 | 1165.04 | 1168.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1167.20 | 1165.04 | 1168.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1168.80 | 1165.04 | 1168.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1166.50 | 1165.33 | 1167.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1152.90 | 1164.04 | 1166.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1167.00 | 1152.69 | 1151.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1167.00 | 1152.69 | 1151.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 1180.00 | 1164.49 | 1158.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 1191.30 | 1195.44 | 1188.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 1193.40 | 1195.44 | 1188.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1192.80 | 1194.91 | 1188.92 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 1179.10 | 1186.30 | 1187.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 1171.00 | 1181.59 | 1184.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1182.60 | 1177.02 | 1181.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 1182.60 | 1177.02 | 1181.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1182.60 | 1177.02 | 1181.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1181.30 | 1177.02 | 1181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1184.80 | 1178.58 | 1181.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 1184.30 | 1178.58 | 1181.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1187.80 | 1180.42 | 1181.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1189.00 | 1180.42 | 1181.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1196.90 | 1183.72 | 1183.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1202.40 | 1187.45 | 1185.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 1188.70 | 1190.15 | 1187.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 1188.70 | 1190.15 | 1187.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1188.70 | 1190.15 | 1187.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 1188.70 | 1190.15 | 1187.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1200.00 | 1192.12 | 1188.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1187.70 | 1192.12 | 1188.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1199.60 | 1200.87 | 1196.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 1199.60 | 1200.87 | 1196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1196.30 | 1199.96 | 1196.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 1194.30 | 1199.96 | 1196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1199.20 | 1199.81 | 1196.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 1203.10 | 1200.48 | 1197.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 1201.00 | 1204.19 | 1200.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1192.50 | 1198.74 | 1198.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1192.50 | 1198.74 | 1198.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1188.50 | 1194.74 | 1196.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1115.00 | 1114.78 | 1132.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 1116.50 | 1114.78 | 1132.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1124.90 | 1118.15 | 1128.42 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1145.00 | 1132.44 | 1130.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 1147.90 | 1137.40 | 1133.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 1135.70 | 1137.06 | 1133.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 1135.70 | 1137.06 | 1133.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1147.10 | 1139.25 | 1135.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1151.20 | 1139.25 | 1135.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 11:30:00 | 1147.80 | 1141.57 | 1137.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 1147.90 | 1141.97 | 1137.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 13:45:00 | 1149.20 | 1142.60 | 1138.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | SL hit (close<static) qty=1.00 sl=1133.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1116.90 | 1133.62 | 1135.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 1101.90 | 1118.72 | 1125.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1094.10 | 1083.31 | 1097.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1094.10 | 1083.31 | 1097.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1094.10 | 1083.31 | 1097.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1093.70 | 1083.31 | 1097.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1098.80 | 1088.80 | 1095.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1098.80 | 1088.80 | 1095.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1104.30 | 1091.90 | 1096.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1107.40 | 1091.90 | 1096.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1094.70 | 1094.98 | 1097.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1097.30 | 1094.98 | 1097.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1096.70 | 1095.17 | 1096.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1096.70 | 1095.17 | 1096.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1099.00 | 1095.93 | 1096.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1099.00 | 1095.93 | 1096.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 14:15:00 | 1107.80 | 1098.31 | 1097.96 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 1088.00 | 1096.53 | 1097.23 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 1113.90 | 1097.31 | 1095.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 1120.00 | 1106.75 | 1100.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1097.80 | 1104.96 | 1100.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1097.80 | 1104.96 | 1100.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1097.80 | 1104.96 | 1100.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1097.80 | 1104.96 | 1100.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1092.00 | 1102.36 | 1099.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1092.30 | 1102.36 | 1099.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1115.80 | 1105.54 | 1102.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 1118.60 | 1107.79 | 1103.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 1116.80 | 1107.79 | 1103.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:45:00 | 1118.80 | 1109.95 | 1104.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 1118.10 | 1135.98 | 1130.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1127.30 | 1130.17 | 1128.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1109.60 | 1130.17 | 1128.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 15:15:00 | 1134.50 | 1126.71 | 1126.28 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1118.00 | 1124.96 | 1125.53 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1131.40 | 1111.41 | 1111.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 14:15:00 | 1137.20 | 1127.94 | 1121.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1118.60 | 1126.72 | 1122.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1118.60 | 1126.72 | 1122.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1118.60 | 1126.72 | 1122.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1116.30 | 1126.72 | 1122.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1122.50 | 1125.88 | 1122.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 1127.90 | 1126.22 | 1122.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1110.70 | 1126.00 | 1124.37 | SL hit (close<static) qty=1.00 sl=1118.60 alert=retest2 |

### Cycle 232 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1107.20 | 1122.24 | 1122.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 1104.30 | 1115.28 | 1119.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 1098.50 | 1097.34 | 1102.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 1098.50 | 1097.34 | 1102.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 1107.00 | 1099.27 | 1103.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 1107.00 | 1099.27 | 1103.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1100.00 | 1099.41 | 1102.82 | EMA400 retest candle locked (from downside) |

### Cycle 233 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1116.40 | 1104.63 | 1104.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 1124.20 | 1111.92 | 1108.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1113.20 | 1118.80 | 1115.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1113.20 | 1118.80 | 1115.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1113.20 | 1118.80 | 1115.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1109.90 | 1118.80 | 1115.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1108.20 | 1116.68 | 1114.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1108.20 | 1116.68 | 1114.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 234 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1099.90 | 1113.33 | 1113.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1095.00 | 1107.66 | 1110.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 1117.30 | 1108.24 | 1109.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 12:15:00 | 1117.30 | 1108.24 | 1109.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1117.30 | 1108.24 | 1109.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:00:00 | 1117.30 | 1108.24 | 1109.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1120.10 | 1110.61 | 1110.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1120.10 | 1110.61 | 1110.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 235 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 1133.20 | 1115.13 | 1112.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1143.90 | 1124.70 | 1117.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 1189.20 | 1194.04 | 1177.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 1189.20 | 1194.04 | 1177.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1195.00 | 1203.01 | 1192.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:45:00 | 1194.70 | 1203.01 | 1192.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1206.10 | 1201.33 | 1194.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 1212.20 | 1203.56 | 1195.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:45:00 | 1221.80 | 1221.79 | 1210.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1171.20 | 1211.67 | 1206.97 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |

### Cycle 236 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1168.90 | 1203.12 | 1203.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1136.70 | 1178.49 | 1190.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1132.40 | 1125.23 | 1142.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1132.40 | 1125.23 | 1142.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1127.70 | 1126.54 | 1140.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1124.00 | 1124.49 | 1138.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 1125.00 | 1130.96 | 1137.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 1123.10 | 1114.76 | 1115.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1145.00 | 1120.81 | 1117.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 237 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1145.00 | 1120.81 | 1117.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 1148.30 | 1126.31 | 1120.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1119.20 | 1134.49 | 1128.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1119.20 | 1134.49 | 1128.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1119.20 | 1134.49 | 1128.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 1117.60 | 1134.49 | 1128.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1127.00 | 1132.99 | 1127.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 1117.90 | 1132.99 | 1127.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1142.80 | 1134.96 | 1129.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 1147.50 | 1134.96 | 1129.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 1145.90 | 1135.60 | 1130.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 1147.00 | 1138.91 | 1132.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1121.20 | 1138.10 | 1133.52 | SL hit (close<static) qty=1.00 sl=1125.00 alert=retest2 |

### Cycle 238 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1111.50 | 1128.86 | 1129.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1101.10 | 1123.31 | 1127.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1073.80 | 1063.95 | 1078.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1073.80 | 1063.95 | 1078.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1073.80 | 1063.95 | 1078.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1045.30 | 1073.20 | 1073.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1085.60 | 1051.49 | 1050.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1085.60 | 1051.49 | 1050.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1088.10 | 1058.81 | 1053.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1071.00 | 1087.87 | 1073.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1071.00 | 1087.87 | 1073.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1071.00 | 1087.87 | 1073.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 1072.20 | 1087.87 | 1073.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1069.20 | 1084.13 | 1073.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1069.20 | 1084.13 | 1073.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1072.70 | 1081.85 | 1073.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:00:00 | 1076.00 | 1078.27 | 1073.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:30:00 | 1075.40 | 1077.68 | 1073.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1075.30 | 1077.68 | 1073.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1065.90 | 1075.32 | 1072.55 | SL hit (close<static) qty=1.00 sl=1066.70 alert=retest2 |

### Cycle 240 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1047.60 | 1069.78 | 1070.28 | EMA200 below EMA400 |

### Cycle 241 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1087.80 | 1071.10 | 1069.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 1090.50 | 1074.98 | 1071.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 1084.00 | 1085.44 | 1078.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 1054.50 | 1085.44 | 1078.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1058.10 | 1079.97 | 1076.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 1051.90 | 1079.97 | 1076.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 1060.10 | 1076.00 | 1074.93 | EMA400 retest candle locked (from upside) |

### Cycle 242 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 1064.80 | 1073.76 | 1074.01 | EMA200 below EMA400 |

### Cycle 243 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1094.10 | 1077.83 | 1075.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1102.40 | 1082.74 | 1078.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 1082.40 | 1088.09 | 1082.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1082.40 | 1088.09 | 1082.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1082.40 | 1088.09 | 1082.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1103.20 | 1088.09 | 1085.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1213.52 | 1178.90 | 1165.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 244 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 1241.80 | 1252.11 | 1252.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 15:15:00 | 1236.70 | 1246.69 | 1249.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1251.50 | 1247.65 | 1249.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1251.50 | 1247.65 | 1249.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1251.50 | 1247.65 | 1249.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1258.20 | 1247.65 | 1249.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1260.70 | 1250.26 | 1250.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1260.70 | 1250.26 | 1250.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 245 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1255.90 | 1251.39 | 1251.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 1269.10 | 1254.93 | 1252.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1257.20 | 1259.02 | 1256.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 10:15:00 | 1257.20 | 1259.02 | 1256.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1257.20 | 1259.02 | 1256.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 1254.00 | 1259.02 | 1256.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1252.40 | 1257.70 | 1255.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1255.60 | 1257.70 | 1255.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1254.50 | 1257.06 | 1255.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 1254.60 | 1257.06 | 1255.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1254.00 | 1256.45 | 1255.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1254.00 | 1256.45 | 1255.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1254.30 | 1256.02 | 1255.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 1253.00 | 1256.02 | 1255.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1253.00 | 1255.41 | 1255.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1267.70 | 1255.41 | 1255.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:45:00 | 1271.80 | 1258.71 | 1256.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1244.80 | 1257.46 | 1257.13 | SL hit (close<static) qty=1.00 sl=1250.00 alert=retest2 |

### Cycle 246 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1246.60 | 1255.29 | 1256.17 | EMA200 below EMA400 |

### Cycle 247 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1281.60 | 1257.04 | 1255.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1305.20 | 1266.67 | 1260.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 13:15:00 | 1267.50 | 1288.10 | 1279.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 13:15:00 | 1267.50 | 1288.10 | 1279.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1267.50 | 1288.10 | 1279.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 1267.50 | 1288.10 | 1279.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1264.70 | 1283.42 | 1278.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 1264.70 | 1283.42 | 1278.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 248 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 1249.00 | 1274.37 | 1274.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 1243.00 | 1268.10 | 1272.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 11:15:00 | 1283.30 | 1271.14 | 1273.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 11:15:00 | 1283.30 | 1271.14 | 1273.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1283.30 | 1271.14 | 1273.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 1283.30 | 1271.14 | 1273.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1280.00 | 1272.91 | 1273.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:30:00 | 1286.40 | 1272.91 | 1273.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 249 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1284.40 | 1276.17 | 1275.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1292.80 | 1279.96 | 1277.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 1291.80 | 1294.78 | 1287.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 1291.80 | 1294.78 | 1287.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1291.80 | 1294.78 | 1287.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1291.80 | 1294.78 | 1287.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1281.20 | 1291.30 | 1287.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1281.20 | 1291.30 | 1287.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1273.60 | 1287.76 | 1286.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 1273.60 | 1287.76 | 1286.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 250 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 1273.10 | 1284.83 | 1284.89 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 1191.65 | 2024-04-15 09:15:00 | 1145.30 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-05-21 09:15:00 | 1283.90 | 2024-05-27 09:15:00 | 1216.70 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2024-05-28 11:30:00 | 1212.05 | 2024-06-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-05-28 12:00:00 | 1209.45 | 2024-06-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-05-28 13:45:00 | 1208.10 | 2024-06-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-05-29 11:00:00 | 1210.20 | 2024-06-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-05-29 15:15:00 | 1210.25 | 2024-06-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-05-30 09:30:00 | 1209.05 | 2024-06-03 09:15:00 | 1236.30 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-06-26 13:15:00 | 1194.10 | 2024-06-27 09:15:00 | 1172.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-06-27 09:15:00 | 1187.35 | 2024-06-27 09:15:00 | 1172.60 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-07-15 12:15:00 | 1342.95 | 2024-07-15 14:15:00 | 1329.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-24 12:30:00 | 1329.15 | 2024-07-24 13:15:00 | 1333.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1333.20 | 2024-07-30 11:15:00 | 1327.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-08-02 10:45:00 | 1345.85 | 2024-08-02 12:15:00 | 1336.75 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-08-12 09:45:00 | 1221.55 | 2024-08-13 12:15:00 | 1248.35 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-08-30 15:00:00 | 1349.30 | 2024-09-09 10:15:00 | 1370.00 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-09-20 09:15:00 | 1360.05 | 2024-09-25 09:15:00 | 1292.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 09:15:00 | 1360.05 | 2024-09-25 12:15:00 | 1323.20 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2024-10-07 09:15:00 | 1332.00 | 2024-10-09 09:15:00 | 1351.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-10-15 12:30:00 | 1298.50 | 2024-10-22 14:15:00 | 1233.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 13:15:00 | 1299.35 | 2024-10-22 14:15:00 | 1234.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 14:15:00 | 1298.80 | 2024-10-22 14:15:00 | 1233.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 1296.45 | 2024-10-22 14:15:00 | 1231.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1290.40 | 2024-10-23 09:15:00 | 1225.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 12:30:00 | 1298.50 | 2024-10-28 09:15:00 | 1168.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-15 13:15:00 | 1299.35 | 2024-10-28 09:15:00 | 1169.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-15 14:15:00 | 1298.80 | 2024-10-28 09:15:00 | 1168.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 1296.45 | 2024-10-28 09:15:00 | 1166.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1290.40 | 2024-10-28 09:15:00 | 1161.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-12 13:15:00 | 1246.85 | 2024-11-13 12:15:00 | 1184.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 13:15:00 | 1246.85 | 2024-11-14 12:15:00 | 1207.90 | STOP_HIT | 0.50 | 3.12% |
| BUY | retest2 | 2024-12-13 09:15:00 | 1303.80 | 2024-12-13 09:15:00 | 1273.35 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-12-13 13:30:00 | 1299.50 | 2024-12-18 15:15:00 | 1310.90 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-12-20 13:15:00 | 1288.05 | 2024-12-26 14:15:00 | 1288.75 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-01-03 12:30:00 | 1284.65 | 2025-01-10 09:15:00 | 1220.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 12:30:00 | 1284.65 | 2025-01-13 13:15:00 | 1156.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 10:45:00 | 1085.45 | 2025-01-27 09:15:00 | 1031.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 11:15:00 | 1085.55 | 2025-01-27 09:15:00 | 1031.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 1084.40 | 2025-01-27 09:15:00 | 1030.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 10:45:00 | 1085.45 | 2025-01-28 10:15:00 | 976.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 11:15:00 | 1085.55 | 2025-01-28 10:15:00 | 977.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 1084.40 | 2025-01-28 14:15:00 | 1018.00 | STOP_HIT | 0.50 | 6.12% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1046.60 | 2025-02-10 09:15:00 | 1023.65 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-02-13 15:15:00 | 975.30 | 2025-02-14 09:15:00 | 926.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 975.30 | 2025-02-14 12:15:00 | 877.77 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 11:30:00 | 938.35 | 2025-04-21 15:15:00 | 1032.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:30:00 | 941.15 | 2025-04-21 15:15:00 | 1035.27 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-07 13:30:00 | 939.60 | 2025-05-07 14:15:00 | 973.65 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-05-27 12:00:00 | 1132.15 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-28 14:30:00 | 1128.35 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1127.80 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-29 09:45:00 | 1129.10 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1115.80 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-03 14:15:00 | 1179.50 | 2025-06-05 15:15:00 | 1151.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-06-04 14:30:00 | 1173.90 | 2025-06-05 15:15:00 | 1151.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-10 12:30:00 | 1172.80 | 2025-06-10 14:15:00 | 1167.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-06-11 09:30:00 | 1174.00 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-11 10:30:00 | 1174.50 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-11 14:00:00 | 1172.60 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-12 09:15:00 | 1189.00 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-06-30 15:00:00 | 1240.80 | 2025-07-01 10:15:00 | 1211.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-03 14:30:00 | 1198.50 | 2025-07-09 09:15:00 | 1196.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-07-04 09:30:00 | 1198.90 | 2025-07-09 09:15:00 | 1196.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-14 11:15:00 | 1188.80 | 2025-07-14 15:15:00 | 1197.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-14 13:00:00 | 1190.20 | 2025-07-14 15:15:00 | 1197.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-14 13:45:00 | 1189.30 | 2025-07-14 15:15:00 | 1197.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1224.00 | 2025-07-17 11:15:00 | 1185.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-07-22 12:15:00 | 1188.60 | 2025-07-23 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-23 10:00:00 | 1189.00 | 2025-07-23 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-06 10:00:00 | 1117.00 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.07% |
| SELL | retest2 | 2025-08-06 10:30:00 | 1116.60 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2025-08-06 12:15:00 | 1115.90 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.18% |
| SELL | retest2 | 2025-08-06 15:00:00 | 1116.10 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.16% |
| SELL | retest2 | 2025-08-07 11:15:00 | 1111.30 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.62% |
| BUY | retest2 | 2025-08-20 10:45:00 | 1276.90 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-20 11:30:00 | 1276.10 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-20 13:30:00 | 1277.80 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-08-21 09:15:00 | 1292.80 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-08-28 15:15:00 | 1264.10 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-29 09:45:00 | 1263.20 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-08-29 10:45:00 | 1263.50 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-08-29 12:30:00 | 1265.00 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-09-01 10:30:00 | 1244.90 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-02 10:00:00 | 1241.90 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-03 09:30:00 | 1245.90 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-09-03 10:45:00 | 1240.00 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-09-05 11:30:00 | 1270.30 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1276.30 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-09-09 09:15:00 | 1273.10 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-09 12:45:00 | 1272.40 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-10 09:30:00 | 1287.00 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-09-10 10:45:00 | 1276.80 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-15 12:15:00 | 1300.80 | 2025-09-16 12:15:00 | 1282.70 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-15 14:00:00 | 1303.00 | 2025-09-16 12:15:00 | 1282.70 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-09-16 11:45:00 | 1301.10 | 2025-09-16 12:15:00 | 1282.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-22 13:00:00 | 1254.60 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-22 13:45:00 | 1252.80 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-23 14:45:00 | 1252.70 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-24 13:30:00 | 1253.60 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-09-29 11:15:00 | 1244.60 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1245.50 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1244.70 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-30 12:45:00 | 1245.60 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-29 14:45:00 | 1264.90 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-29 15:15:00 | 1265.00 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-30 10:15:00 | 1262.40 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-30 13:00:00 | 1261.10 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1291.00 | 2025-11-10 09:15:00 | 1284.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-12 15:15:00 | 1260.10 | 2025-11-24 10:15:00 | 1197.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 09:45:00 | 1260.20 | 2025-11-24 10:15:00 | 1197.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1255.40 | 2025-11-24 14:15:00 | 1192.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 15:15:00 | 1260.10 | 2025-11-25 09:15:00 | 1220.60 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-11-13 09:45:00 | 1260.20 | 2025-11-25 09:15:00 | 1220.60 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1255.40 | 2025-11-25 09:15:00 | 1220.60 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-12-03 10:15:00 | 1171.90 | 2025-12-04 11:15:00 | 1194.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1161.70 | 2025-12-09 14:15:00 | 1171.90 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-08 10:45:00 | 1155.50 | 2025-12-09 14:15:00 | 1171.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1173.30 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-11 09:45:00 | 1173.80 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-11 10:15:00 | 1177.90 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-12 09:15:00 | 1173.20 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1152.90 | 2025-12-22 10:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-05 09:30:00 | 1203.10 | 2026-01-06 11:15:00 | 1192.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-05 14:15:00 | 1201.00 | 2026-01-06 11:15:00 | 1192.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-16 10:15:00 | 1151.20 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-01-16 11:30:00 | 1147.80 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-16 12:45:00 | 1147.90 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-01-16 13:45:00 | 1149.20 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-01-30 10:30:00 | 1118.60 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-01-30 11:00:00 | 1116.80 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-01-30 11:45:00 | 1118.80 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-02-02 11:30:00 | 1118.10 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-11 11:45:00 | 1127.90 | 2026-02-12 09:15:00 | 1110.70 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-02-27 10:45:00 | 1212.20 | 2026-03-02 10:15:00 | 1171.20 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-03-02 09:45:00 | 1221.80 | 2026-03-02 10:15:00 | 1171.20 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2026-03-06 10:30:00 | 1124.00 | 2026-03-11 10:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1125.00 | 2026-03-11 10:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-11 10:15:00 | 1123.10 | 2026-03-11 10:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-03-12 12:15:00 | 1147.50 | 2026-03-13 09:15:00 | 1121.20 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-12 13:15:00 | 1145.90 | 2026-03-13 09:15:00 | 1121.20 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-03-12 15:00:00 | 1147.00 | 2026-03-13 09:15:00 | 1121.20 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1045.30 | 2026-03-25 09:15:00 | 1085.60 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2026-03-27 14:00:00 | 1076.00 | 2026-03-27 15:15:00 | 1065.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-03-27 14:30:00 | 1075.40 | 2026-03-27 15:15:00 | 1065.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-03-27 15:00:00 | 1075.30 | 2026-03-27 15:15:00 | 1065.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1103.20 | 2026-04-15 09:15:00 | 1213.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 1267.70 | 2026-04-30 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-29 12:45:00 | 1271.80 | 2026-04-30 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -2.12% |
