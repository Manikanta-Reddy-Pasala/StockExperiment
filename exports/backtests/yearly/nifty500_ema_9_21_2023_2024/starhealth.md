# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 519.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 209 |
| ALERT1 | 130 |
| ALERT2 | 129 |
| ALERT2_SKIP | 89 |
| ALERT3 | 310 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 149 |
| PARTIAL | 27 |
| TARGET_HIT | 9 |
| STOP_HIT | 142 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 178 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 86 / 92
- **Target hits / Stop hits / Partials:** 9 / 142 / 27
- **Avg / median % per leg:** 1.31% / -0.17%
- **Sum % (uncompounded):** 233.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 19 | 31.7% | 9 | 50 | 1 | 1.02% | 61.3% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.49% | 9.0% |
| BUY @ 3rd Alert (retest2) | 58 | 17 | 29.3% | 9 | 49 | 0 | 0.90% | 52.3% |
| SELL (all) | 118 | 67 | 56.8% | 0 | 92 | 26 | 1.46% | 172.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.97% | -3.0% |
| SELL @ 3rd Alert (retest2) | 117 | 67 | 57.3% | 0 | 91 | 26 | 1.50% | 175.5% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.01% | 6.0% |
| retest2 (combined) | 175 | 84 | 48.0% | 9 | 140 | 26 | 1.30% | 227.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 10:15:00 | 598.45 | 585.72 | 585.53 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 12:15:00 | 583.00 | 586.96 | 587.24 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 590.20 | 587.56 | 587.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 12:15:00 | 595.10 | 589.07 | 587.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 588.10 | 590.71 | 589.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 588.10 | 590.71 | 589.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 588.10 | 590.71 | 589.30 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 14:15:00 | 585.95 | 588.28 | 588.50 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 595.80 | 588.35 | 587.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 599.00 | 590.48 | 588.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 09:15:00 | 587.55 | 591.70 | 590.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 587.55 | 591.70 | 590.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 587.55 | 591.70 | 590.52 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 584.85 | 589.23 | 589.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 13:15:00 | 582.65 | 585.87 | 587.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 13:15:00 | 544.30 | 543.04 | 555.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 540.70 | 539.99 | 543.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 540.70 | 539.99 | 543.59 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 12:15:00 | 542.35 | 540.06 | 539.80 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 14:15:00 | 539.70 | 540.25 | 540.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 09:15:00 | 534.90 | 538.44 | 539.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 516.75 | 513.05 | 518.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 516.55 | 513.75 | 518.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 516.55 | 513.75 | 518.23 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 526.00 | 519.45 | 519.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 530.30 | 525.57 | 523.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 536.00 | 538.30 | 534.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 536.00 | 538.30 | 534.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 536.00 | 538.30 | 534.08 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 13:15:00 | 540.45 | 546.93 | 546.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 14:15:00 | 536.00 | 544.75 | 545.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 550.80 | 545.15 | 545.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 550.80 | 545.15 | 545.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 550.80 | 545.15 | 545.90 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 10:15:00 | 553.00 | 546.72 | 546.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 559.50 | 551.84 | 549.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 11:15:00 | 590.30 | 590.54 | 583.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 15:15:00 | 590.85 | 590.35 | 585.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 590.85 | 590.35 | 585.43 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 09:15:00 | 642.95 | 644.98 | 645.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 624.80 | 633.78 | 637.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 10:15:00 | 626.00 | 624.04 | 629.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 11:15:00 | 630.70 | 625.38 | 629.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 630.70 | 625.38 | 629.48 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 642.75 | 633.07 | 631.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 10:15:00 | 651.60 | 636.78 | 633.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 15:15:00 | 639.80 | 640.72 | 637.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 647.00 | 641.97 | 638.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 647.00 | 641.97 | 638.12 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-08-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 10:15:00 | 632.15 | 637.70 | 637.94 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 640.60 | 638.29 | 638.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 647.85 | 640.15 | 638.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 11:15:00 | 638.55 | 640.17 | 639.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 11:15:00 | 638.55 | 640.17 | 639.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 638.55 | 640.17 | 639.19 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 632.85 | 640.07 | 640.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 15:15:00 | 628.65 | 635.60 | 637.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 637.00 | 635.88 | 637.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 637.00 | 635.88 | 637.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 637.00 | 635.88 | 637.70 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 11:15:00 | 638.50 | 634.94 | 634.92 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 14:15:00 | 627.75 | 634.41 | 635.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 12:15:00 | 625.25 | 630.68 | 632.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 12:15:00 | 624.90 | 624.51 | 627.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 626.15 | 624.45 | 626.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 626.15 | 624.45 | 626.75 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 629.00 | 617.29 | 616.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 634.90 | 627.49 | 622.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 15:15:00 | 627.00 | 628.19 | 624.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 630.20 | 628.59 | 625.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 630.20 | 628.59 | 625.31 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 652.20 | 658.00 | 658.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 640.55 | 652.13 | 655.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 642.10 | 637.41 | 642.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 642.10 | 637.41 | 642.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 642.10 | 637.41 | 642.88 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 14:15:00 | 645.75 | 641.44 | 641.29 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 637.00 | 640.69 | 641.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 630.95 | 637.00 | 639.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 585.00 | 583.68 | 593.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 585.00 | 583.68 | 593.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 585.00 | 583.68 | 593.24 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 608.15 | 594.53 | 593.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 15:15:00 | 612.50 | 598.12 | 594.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 10:15:00 | 597.05 | 598.56 | 595.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 601.65 | 601.04 | 598.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 601.65 | 601.04 | 598.25 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 593.65 | 597.76 | 598.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 587.90 | 594.12 | 596.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 12:15:00 | 593.45 | 593.21 | 595.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 13:15:00 | 587.90 | 592.15 | 594.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 587.90 | 592.15 | 594.45 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 13:15:00 | 595.25 | 593.79 | 593.66 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 587.00 | 592.86 | 593.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 581.75 | 587.87 | 590.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 586.35 | 581.33 | 584.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 586.35 | 581.33 | 584.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 586.35 | 581.33 | 584.60 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 12:15:00 | 588.60 | 584.05 | 583.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 595.75 | 590.01 | 588.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 09:15:00 | 593.10 | 594.03 | 591.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 593.10 | 594.03 | 591.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 593.10 | 594.03 | 591.68 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 588.80 | 592.61 | 592.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 586.45 | 591.37 | 592.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 15:15:00 | 592.00 | 590.93 | 591.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 15:15:00 | 592.00 | 590.93 | 591.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 592.00 | 590.93 | 591.70 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 591.90 | 585.28 | 585.26 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 11:15:00 | 583.75 | 586.39 | 586.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 12:15:00 | 582.00 | 585.51 | 586.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 10:15:00 | 553.90 | 551.18 | 557.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 10:15:00 | 553.90 | 551.18 | 557.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 553.90 | 551.18 | 557.58 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 551.70 | 545.37 | 545.00 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 10:15:00 | 544.90 | 546.31 | 546.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 12:15:00 | 543.60 | 544.70 | 545.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 15:15:00 | 545.00 | 544.75 | 545.17 | EMA200 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 549.60 | 545.72 | 545.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 560.20 | 551.25 | 548.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 13:15:00 | 559.95 | 561.95 | 558.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 14:15:00 | 560.55 | 561.67 | 558.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 560.55 | 561.67 | 558.49 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 13:15:00 | 566.30 | 569.42 | 569.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 14:15:00 | 560.45 | 567.63 | 568.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 566.80 | 566.31 | 567.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 09:15:00 | 566.80 | 566.31 | 567.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 566.80 | 566.31 | 567.80 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 565.10 | 560.85 | 560.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 13:15:00 | 569.95 | 562.67 | 561.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 565.75 | 566.06 | 563.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 10:15:00 | 565.75 | 566.06 | 563.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 565.75 | 566.06 | 563.82 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 560.15 | 563.16 | 563.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 10:15:00 | 555.20 | 561.57 | 562.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 556.50 | 555.69 | 558.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 556.50 | 555.69 | 558.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 556.50 | 555.69 | 558.55 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 12:15:00 | 529.00 | 525.75 | 525.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 532.50 | 527.64 | 526.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 532.10 | 533.79 | 531.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 532.10 | 533.79 | 531.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 532.10 | 533.79 | 531.25 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 548.55 | 550.41 | 550.57 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 554.05 | 550.65 | 550.31 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 549.45 | 550.11 | 550.14 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 10:15:00 | 550.45 | 550.18 | 550.16 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 550.00 | 550.15 | 550.15 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 12:15:00 | 551.95 | 550.51 | 550.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 15:15:00 | 553.50 | 551.41 | 550.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 09:15:00 | 550.35 | 551.20 | 550.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 550.35 | 551.20 | 550.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 550.35 | 551.20 | 550.75 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 561.00 | 564.31 | 564.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 558.80 | 563.00 | 563.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 563.00 | 562.58 | 563.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 556.20 | 561.31 | 562.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 556.20 | 561.31 | 562.79 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 10:15:00 | 565.10 | 559.37 | 558.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 579.25 | 566.62 | 562.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 575.10 | 575.20 | 570.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 575.00 | 579.97 | 575.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 575.00 | 579.97 | 575.72 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 572.00 | 574.35 | 574.36 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 579.15 | 575.00 | 574.55 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 15:15:00 | 570.00 | 574.31 | 574.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 13:15:00 | 564.95 | 570.01 | 571.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 560.15 | 558.48 | 562.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 11:15:00 | 559.15 | 558.66 | 562.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 559.15 | 558.66 | 562.22 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 558.00 | 552.86 | 552.71 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 10:15:00 | 549.90 | 553.07 | 553.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 13:15:00 | 547.70 | 551.04 | 552.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 15:15:00 | 553.50 | 551.39 | 552.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 15:15:00 | 553.50 | 551.39 | 552.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 553.50 | 551.39 | 552.13 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 09:15:00 | 561.50 | 553.41 | 552.98 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 555.00 | 562.31 | 562.72 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 14:15:00 | 564.25 | 561.34 | 561.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 570.00 | 563.50 | 562.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 573.20 | 576.94 | 573.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 10:15:00 | 573.20 | 576.94 | 573.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 573.20 | 576.94 | 573.95 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 569.60 | 572.30 | 572.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 568.50 | 571.54 | 572.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 569.80 | 569.34 | 570.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 560.00 | 564.51 | 567.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 560.00 | 564.51 | 567.75 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 13:15:00 | 555.20 | 551.62 | 551.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 14:15:00 | 556.00 | 552.49 | 551.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 09:15:00 | 550.95 | 552.91 | 552.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 550.95 | 552.91 | 552.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 550.95 | 552.91 | 552.29 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 546.00 | 551.07 | 551.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 540.50 | 548.36 | 549.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 14:15:00 | 545.25 | 543.92 | 546.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 15:15:00 | 545.50 | 544.24 | 546.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 545.50 | 544.24 | 546.62 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 549.90 | 547.03 | 546.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 14:15:00 | 558.00 | 549.41 | 548.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 11:15:00 | 549.05 | 550.19 | 548.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 11:15:00 | 549.05 | 550.19 | 548.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 549.05 | 550.19 | 548.94 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 526.80 | 546.68 | 548.00 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 548.45 | 545.29 | 545.28 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 541.45 | 544.52 | 544.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 15:15:00 | 535.95 | 542.81 | 544.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 09:15:00 | 542.75 | 541.41 | 542.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 09:15:00 | 542.75 | 541.41 | 542.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 542.75 | 541.41 | 542.49 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 10:15:00 | 548.80 | 542.73 | 542.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 12:15:00 | 549.70 | 544.95 | 543.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 10:15:00 | 547.05 | 548.50 | 546.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 11:15:00 | 546.55 | 548.11 | 546.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 546.55 | 548.11 | 546.05 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 554.95 | 557.56 | 557.71 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 09:15:00 | 567.70 | 559.53 | 558.56 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 560.00 | 562.14 | 562.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 10:15:00 | 558.25 | 560.82 | 561.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 12:15:00 | 560.50 | 560.30 | 561.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-12 09:15:00 | 560.85 | 559.61 | 560.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 559.00 | 559.49 | 560.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 11:45:00 | 556.60 | 559.00 | 560.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:15:00 | 556.40 | 559.00 | 560.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 12:15:00 | 560.15 | 556.28 | 555.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 12:15:00 | 560.15 | 556.28 | 555.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 13:15:00 | 565.05 | 559.12 | 557.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 09:15:00 | 568.10 | 573.17 | 569.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 568.10 | 573.17 | 569.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 568.10 | 573.17 | 569.69 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 566.60 | 568.28 | 568.35 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 570.00 | 568.62 | 568.50 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 10:15:00 | 566.50 | 569.16 | 569.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 11:15:00 | 563.05 | 567.94 | 568.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 09:15:00 | 561.95 | 561.35 | 563.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 561.95 | 561.35 | 563.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 561.95 | 561.35 | 563.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 10:15:00 | 560.05 | 561.35 | 563.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 12:15:00 | 567.00 | 562.92 | 563.77 | SL hit (close>static) qty=1.00 sl=565.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 569.20 | 564.69 | 564.26 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 13:15:00 | 564.00 | 565.41 | 565.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 562.55 | 564.84 | 565.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 559.90 | 558.01 | 560.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 11:00:00 | 559.90 | 558.01 | 560.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 561.35 | 558.68 | 560.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:00:00 | 558.25 | 558.59 | 560.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 530.34 | 536.83 | 539.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 11:15:00 | 537.85 | 536.44 | 539.16 | SL hit (close>ema200) qty=0.50 sl=536.44 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 538.00 | 533.78 | 533.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 538.30 | 534.69 | 533.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 552.00 | 552.36 | 548.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 10:15:00 | 549.15 | 551.72 | 548.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 549.15 | 551.72 | 548.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 549.35 | 551.72 | 548.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 545.70 | 550.51 | 548.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 545.70 | 550.51 | 548.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 544.90 | 549.39 | 547.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:30:00 | 544.45 | 549.39 | 547.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 544.75 | 546.87 | 547.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 537.40 | 544.98 | 546.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 551.55 | 542.84 | 543.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 551.55 | 542.84 | 543.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 551.55 | 542.84 | 543.82 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 552.00 | 544.67 | 544.57 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 544.70 | 545.28 | 545.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 534.50 | 543.08 | 544.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 15:15:00 | 535.05 | 531.79 | 535.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 15:15:00 | 535.05 | 531.79 | 535.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 535.05 | 531.79 | 535.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:30:00 | 521.60 | 528.25 | 533.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 519.10 | 518.27 | 521.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 09:30:00 | 521.40 | 518.41 | 520.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 495.52 | 509.10 | 513.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 493.14 | 509.10 | 513.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 495.33 | 509.10 | 513.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 503.55 | 501.91 | 505.59 | SL hit (close>ema200) qty=0.50 sl=501.91 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 12:15:00 | 512.80 | 503.35 | 502.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 515.95 | 508.76 | 505.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 518.95 | 519.85 | 515.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 14:00:00 | 518.95 | 519.85 | 515.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 515.50 | 517.55 | 516.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 515.50 | 517.55 | 516.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 517.00 | 517.44 | 516.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:15:00 | 516.20 | 517.44 | 516.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 516.20 | 517.19 | 516.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 518.00 | 517.19 | 516.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 10:15:00 | 519.00 | 525.25 | 525.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 519.00 | 525.25 | 525.90 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 524.70 | 523.34 | 523.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 526.35 | 524.35 | 523.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 524.25 | 524.68 | 524.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 524.25 | 524.68 | 524.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 524.25 | 524.68 | 524.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 524.25 | 524.68 | 524.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 524.90 | 524.73 | 524.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:30:00 | 525.85 | 524.73 | 524.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 524.60 | 526.07 | 525.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 524.60 | 526.07 | 525.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 522.25 | 525.31 | 524.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 522.25 | 525.31 | 524.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 563.55 | 566.46 | 558.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 558.80 | 566.46 | 558.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 583.95 | 576.67 | 570.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 586.75 | 578.21 | 571.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 586.20 | 578.21 | 571.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 589.00 | 580.28 | 572.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 572.25 | 578.12 | 578.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 572.25 | 578.12 | 578.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 566.80 | 572.83 | 575.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 12:15:00 | 574.65 | 573.04 | 574.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 12:15:00 | 574.65 | 573.04 | 574.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 574.65 | 573.04 | 574.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:45:00 | 575.00 | 573.04 | 574.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 572.90 | 573.01 | 574.53 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 578.90 | 575.30 | 575.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 14:15:00 | 579.40 | 576.63 | 575.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 585.50 | 585.55 | 583.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 585.50 | 585.55 | 583.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 585.50 | 585.55 | 583.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:45:00 | 583.90 | 585.55 | 583.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 591.00 | 592.63 | 589.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 591.00 | 592.63 | 589.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 587.00 | 591.51 | 588.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 587.70 | 591.51 | 588.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 587.95 | 590.80 | 588.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:30:00 | 586.80 | 590.80 | 588.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 586.55 | 589.95 | 588.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 591.60 | 589.95 | 588.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:00:00 | 590.00 | 589.96 | 588.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 15:15:00 | 584.55 | 589.10 | 589.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 584.55 | 589.10 | 589.11 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 590.55 | 589.39 | 589.24 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 587.65 | 588.89 | 589.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 14:15:00 | 582.15 | 587.54 | 588.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 590.15 | 586.95 | 587.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 10:15:00 | 590.15 | 586.95 | 587.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 590.15 | 586.95 | 587.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 590.15 | 586.95 | 587.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 585.85 | 586.73 | 587.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 574.10 | 586.73 | 587.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 581.25 | 585.64 | 587.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 583.05 | 586.21 | 587.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 595.00 | 587.97 | 587.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 595.00 | 587.97 | 587.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 609.70 | 593.27 | 590.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 614.00 | 617.49 | 612.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 614.00 | 617.49 | 612.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 613.00 | 616.59 | 612.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 620.95 | 616.59 | 612.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 10:15:00 | 610.65 | 615.38 | 612.44 | SL hit (close<static) qty=1.00 sl=611.30 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 608.50 | 613.28 | 613.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 604.80 | 611.58 | 613.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 612.30 | 610.22 | 611.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 612.30 | 610.22 | 611.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 612.30 | 610.22 | 611.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 612.50 | 610.22 | 611.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 610.50 | 610.28 | 611.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:30:00 | 608.10 | 609.13 | 611.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 577.70 | 590.34 | 598.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 577.95 | 574.02 | 578.98 | SL hit (close>ema200) qty=0.50 sl=574.02 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 594.35 | 583.17 | 581.73 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 577.65 | 582.06 | 582.51 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 589.15 | 582.61 | 582.17 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 580.90 | 582.12 | 582.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 571.55 | 578.86 | 580.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 577.30 | 574.86 | 576.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 577.30 | 574.86 | 576.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 577.30 | 574.86 | 576.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 577.30 | 574.86 | 576.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 577.40 | 575.37 | 576.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:15:00 | 577.45 | 575.37 | 576.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 576.90 | 575.68 | 576.80 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 582.70 | 578.15 | 577.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 584.45 | 579.41 | 578.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 597.80 | 607.40 | 602.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 597.80 | 607.40 | 602.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 597.80 | 607.40 | 602.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 597.80 | 607.40 | 602.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 597.10 | 605.34 | 602.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 597.35 | 605.34 | 602.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 601.50 | 603.26 | 601.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:30:00 | 601.60 | 603.26 | 601.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 602.30 | 603.07 | 601.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 608.25 | 602.85 | 601.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:30:00 | 604.00 | 603.60 | 602.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:30:00 | 605.00 | 603.81 | 602.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 12:15:00 | 602.95 | 608.68 | 609.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 602.95 | 608.68 | 609.32 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 617.40 | 609.41 | 609.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 626.00 | 612.73 | 610.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 620.65 | 620.89 | 616.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 620.65 | 620.89 | 616.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 620.25 | 620.14 | 617.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 15:15:00 | 622.00 | 620.14 | 617.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 13:15:00 | 629.80 | 633.89 | 634.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 629.80 | 633.89 | 634.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 14:15:00 | 625.25 | 632.16 | 633.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 11:15:00 | 623.15 | 622.33 | 625.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 12:00:00 | 623.15 | 622.33 | 625.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 623.95 | 622.65 | 625.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:45:00 | 624.80 | 622.65 | 625.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 619.15 | 621.95 | 624.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 10:45:00 | 615.45 | 619.91 | 623.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:45:00 | 616.05 | 616.13 | 619.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 12:00:00 | 617.00 | 616.45 | 618.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 10:45:00 | 617.40 | 615.58 | 615.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 601.75 | 606.50 | 610.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 601.75 | 606.50 | 610.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 615.00 | 607.40 | 609.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 615.15 | 607.40 | 609.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 613.70 | 608.66 | 610.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 613.70 | 608.66 | 610.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 605.45 | 607.17 | 608.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:30:00 | 608.95 | 607.17 | 608.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 609.50 | 607.45 | 608.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 610.60 | 607.45 | 608.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 608.35 | 607.63 | 608.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-20 13:15:00 | 613.20 | 609.77 | 609.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 613.20 | 609.77 | 609.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 616.70 | 611.16 | 610.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 611.85 | 612.23 | 610.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 611.85 | 612.23 | 610.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 611.85 | 612.23 | 610.87 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 607.10 | 609.78 | 610.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 601.25 | 604.74 | 606.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 605.30 | 602.76 | 604.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 605.30 | 602.76 | 604.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 605.30 | 602.76 | 604.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 605.30 | 602.76 | 604.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 604.10 | 603.03 | 604.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 607.45 | 603.03 | 604.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 607.30 | 603.88 | 605.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 607.30 | 603.88 | 605.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 609.85 | 605.07 | 605.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 609.85 | 605.07 | 605.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 610.75 | 606.21 | 605.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 614.45 | 609.39 | 607.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 611.50 | 611.75 | 609.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 12:45:00 | 611.25 | 611.75 | 609.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 610.50 | 611.50 | 609.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:30:00 | 608.00 | 611.50 | 609.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 616.05 | 612.41 | 610.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:45:00 | 608.55 | 612.41 | 610.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 610.00 | 611.93 | 610.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 606.05 | 611.93 | 610.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 603.20 | 610.18 | 609.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 603.20 | 610.18 | 609.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 603.65 | 608.88 | 608.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 594.35 | 602.09 | 604.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 571.70 | 569.33 | 575.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 571.70 | 569.33 | 575.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 576.15 | 570.35 | 574.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 575.60 | 570.35 | 574.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 575.10 | 571.30 | 574.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 576.00 | 571.30 | 574.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 571.45 | 571.33 | 574.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 563.90 | 574.21 | 574.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 562.15 | 557.81 | 557.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 562.15 | 557.81 | 557.43 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 552.95 | 556.85 | 557.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 550.65 | 555.61 | 556.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 15:15:00 | 551.50 | 550.80 | 552.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-18 09:15:00 | 540.50 | 550.80 | 552.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 550.40 | 549.94 | 551.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 550.80 | 549.94 | 551.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 550.00 | 549.95 | 551.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 556.80 | 549.95 | 551.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 556.55 | 551.27 | 551.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 556.55 | 551.27 | 551.61 | SL hit (close>ema400) qty=1.00 sl=551.61 alert=retest1 |

### Cycle 99 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 549.75 | 547.78 | 547.64 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 546.55 | 547.53 | 547.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 544.25 | 546.88 | 547.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 15:15:00 | 549.00 | 547.30 | 547.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 15:15:00 | 549.00 | 547.30 | 547.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 549.00 | 547.30 | 547.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 550.00 | 547.30 | 547.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 546.05 | 547.05 | 547.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 10:30:00 | 544.60 | 546.52 | 547.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 545.40 | 546.59 | 546.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:00:00 | 545.55 | 546.38 | 546.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 537.30 | 543.41 | 545.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 543.50 | 539.86 | 542.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 534.15 | 539.86 | 542.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:15:00 | 538.20 | 538.51 | 540.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:15:00 | 538.30 | 538.70 | 540.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:15:00 | 538.20 | 539.53 | 540.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 540.05 | 539.64 | 540.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:30:00 | 531.05 | 539.48 | 540.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 15:00:00 | 535.15 | 538.61 | 539.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 517.37 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 518.13 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 518.27 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 510.43 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 507.44 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 511.29 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 511.38 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 511.29 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 504.50 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 508.39 | 531.78 | 536.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-01 17:15:00 | 513.95 | 509.25 | 516.61 | SL hit (close>ema200) qty=0.50 sl=509.25 alert=retest2 |

### Cycle 101 — BUY (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 11:15:00 | 498.55 | 488.37 | 488.15 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 484.25 | 490.17 | 490.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 482.20 | 488.58 | 489.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 481.45 | 476.66 | 481.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 14:15:00 | 481.45 | 476.66 | 481.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 481.45 | 476.66 | 481.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:00:00 | 481.45 | 476.66 | 481.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 473.00 | 475.93 | 480.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 470.95 | 474.88 | 479.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:00:00 | 470.70 | 474.88 | 479.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:30:00 | 471.05 | 473.63 | 477.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 471.05 | 473.18 | 476.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 468.30 | 471.35 | 475.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:30:00 | 471.85 | 471.35 | 475.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 469.30 | 468.69 | 471.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 465.45 | 472.63 | 472.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:45:00 | 465.30 | 469.62 | 471.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 466.40 | 468.90 | 470.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 464.40 | 461.38 | 461.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 464.40 | 461.38 | 461.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 13:15:00 | 465.55 | 462.44 | 461.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 462.95 | 465.62 | 463.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 462.95 | 465.62 | 463.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 462.95 | 465.62 | 463.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 462.20 | 465.62 | 463.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 462.00 | 464.90 | 463.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:00:00 | 462.00 | 464.90 | 463.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 458.25 | 461.95 | 462.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 11:15:00 | 456.45 | 459.84 | 461.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 459.70 | 458.23 | 459.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 459.70 | 458.23 | 459.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 459.70 | 458.23 | 459.69 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 15:15:00 | 462.85 | 460.40 | 460.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 471.65 | 464.01 | 462.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 13:15:00 | 466.75 | 467.02 | 464.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 14:00:00 | 466.75 | 467.02 | 464.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 487.60 | 490.31 | 488.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 487.60 | 490.31 | 488.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 490.00 | 490.25 | 488.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 478.40 | 490.25 | 488.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 471.35 | 486.47 | 487.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 10:15:00 | 467.90 | 482.75 | 485.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 11:15:00 | 460.35 | 460.09 | 464.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 12:00:00 | 460.35 | 460.09 | 464.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 463.95 | 460.86 | 464.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:30:00 | 461.90 | 460.86 | 464.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 464.70 | 461.63 | 464.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:45:00 | 464.00 | 461.63 | 464.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 463.25 | 461.95 | 464.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:00:00 | 462.05 | 462.29 | 463.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 462.70 | 462.49 | 463.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 12:30:00 | 462.20 | 462.74 | 463.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 13:15:00 | 462.50 | 462.74 | 463.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 463.20 | 462.83 | 463.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 463.20 | 462.83 | 463.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 464.60 | 463.19 | 463.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 464.60 | 463.19 | 463.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 464.95 | 463.54 | 463.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 467.25 | 463.54 | 463.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 467.85 | 464.40 | 464.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 467.85 | 464.40 | 464.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 469.95 | 465.69 | 464.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 480.00 | 484.62 | 480.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 480.00 | 484.62 | 480.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 480.00 | 484.62 | 480.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:30:00 | 484.70 | 484.13 | 480.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:15:00 | 484.30 | 484.13 | 480.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 475.65 | 485.10 | 484.82 | SL hit (close<static) qty=1.00 sl=476.45 alert=retest2 |

### Cycle 108 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 475.80 | 483.24 | 484.00 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 488.30 | 482.15 | 482.15 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 478.20 | 483.40 | 483.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 15:15:00 | 476.00 | 480.23 | 481.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 479.70 | 478.52 | 480.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 479.70 | 478.52 | 480.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 479.70 | 478.52 | 480.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 479.70 | 478.52 | 480.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 479.45 | 478.70 | 480.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 477.10 | 478.70 | 480.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:00:00 | 478.00 | 478.03 | 479.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 476.95 | 478.42 | 479.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 476.15 | 478.42 | 479.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 476.00 | 477.94 | 479.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 474.85 | 477.48 | 478.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 485.75 | 478.71 | 479.12 | SL hit (close>static) qty=1.00 sl=480.75 alert=retest2 |

### Cycle 111 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 485.80 | 480.13 | 479.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 488.65 | 484.60 | 482.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 485.00 | 487.43 | 485.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 12:15:00 | 485.00 | 487.43 | 485.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 485.00 | 487.43 | 485.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 485.00 | 487.43 | 485.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 484.00 | 486.74 | 485.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 484.00 | 486.74 | 485.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 482.90 | 485.98 | 485.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 482.90 | 485.98 | 485.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 483.75 | 485.37 | 485.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 483.75 | 485.37 | 485.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 481.05 | 484.51 | 484.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 480.25 | 483.79 | 484.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 481.40 | 477.03 | 479.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 14:15:00 | 481.40 | 477.03 | 479.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 481.40 | 477.03 | 479.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 481.40 | 477.03 | 479.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 483.00 | 478.23 | 479.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 474.20 | 478.23 | 479.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 12:15:00 | 468.40 | 466.36 | 466.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 468.40 | 466.36 | 466.29 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 463.45 | 466.02 | 466.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 462.20 | 464.34 | 465.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 462.65 | 461.53 | 463.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 12:15:00 | 462.65 | 461.53 | 463.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 462.65 | 461.53 | 463.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 462.65 | 461.53 | 463.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 462.50 | 461.73 | 463.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:45:00 | 462.30 | 461.73 | 463.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 470.00 | 463.25 | 463.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 469.35 | 463.25 | 463.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 461.60 | 462.92 | 463.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 458.20 | 463.56 | 463.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:15:00 | 460.45 | 463.23 | 463.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:45:00 | 460.00 | 462.53 | 463.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 457.05 | 459.82 | 461.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 468.00 | 461.46 | 461.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 468.35 | 461.46 | 461.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 463.55 | 461.87 | 461.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:00:00 | 462.45 | 461.99 | 462.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:00:00 | 462.10 | 462.01 | 462.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 13:15:00 | 463.75 | 462.36 | 462.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 463.75 | 462.36 | 462.18 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 458.95 | 461.55 | 461.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 458.15 | 460.61 | 461.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 445.00 | 443.42 | 448.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:00:00 | 445.00 | 443.42 | 448.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 442.10 | 440.96 | 444.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 442.50 | 440.96 | 444.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 439.00 | 437.33 | 439.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:45:00 | 442.00 | 437.33 | 439.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 441.50 | 436.20 | 437.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 440.50 | 436.20 | 437.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 450.50 | 439.06 | 438.86 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 436.00 | 440.93 | 441.12 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 443.15 | 440.61 | 440.58 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 439.10 | 440.43 | 440.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 437.50 | 439.40 | 440.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 432.75 | 432.33 | 435.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 432.75 | 432.33 | 435.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 436.90 | 433.24 | 435.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 428.90 | 433.24 | 435.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:15:00 | 407.45 | 416.87 | 424.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 403.20 | 396.69 | 404.54 | SL hit (close>ema200) qty=0.50 sl=396.69 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 13:15:00 | 374.15 | 370.52 | 370.37 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 364.95 | 370.30 | 370.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 12:15:00 | 358.00 | 364.56 | 366.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 364.20 | 361.69 | 364.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 10:15:00 | 364.20 | 361.69 | 364.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 364.20 | 361.69 | 364.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 364.20 | 361.69 | 364.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 363.65 | 362.08 | 364.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 364.00 | 362.08 | 364.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 363.25 | 362.31 | 364.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 364.75 | 362.31 | 364.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 365.25 | 362.90 | 364.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:45:00 | 365.90 | 362.90 | 364.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 366.80 | 363.68 | 364.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 366.80 | 363.68 | 364.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 365.75 | 364.10 | 364.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 372.90 | 364.10 | 364.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 374.40 | 366.16 | 365.43 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 364.80 | 368.02 | 368.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 358.15 | 365.44 | 366.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 365.05 | 363.06 | 365.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 365.05 | 363.06 | 365.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 365.05 | 363.06 | 365.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 365.05 | 363.06 | 365.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 363.00 | 363.04 | 364.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 363.00 | 363.04 | 364.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 365.00 | 363.44 | 364.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 363.00 | 363.44 | 364.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 358.40 | 362.43 | 364.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 356.80 | 361.65 | 363.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 355.25 | 361.65 | 363.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 357.65 | 356.73 | 359.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 356.70 | 357.22 | 359.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 359.25 | 356.31 | 358.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 362.20 | 356.31 | 358.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 357.00 | 356.45 | 357.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 358.85 | 356.45 | 357.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 357.30 | 356.62 | 357.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 357.10 | 356.62 | 357.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 356.00 | 356.49 | 357.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:00:00 | 354.80 | 356.23 | 357.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 354.50 | 355.95 | 356.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 15:00:00 | 354.55 | 355.33 | 356.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 359.35 | 356.03 | 356.56 | SL hit (close>static) qty=1.00 sl=358.20 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 10:15:00 | 360.25 | 356.53 | 356.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 371.25 | 361.36 | 358.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 362.75 | 362.92 | 360.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 13:15:00 | 362.75 | 362.92 | 360.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 362.75 | 362.92 | 360.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 361.75 | 362.92 | 360.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 359.75 | 362.87 | 361.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:45:00 | 359.60 | 362.87 | 361.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 360.00 | 362.30 | 361.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:00:00 | 360.00 | 362.30 | 361.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 15:15:00 | 358.80 | 360.56 | 360.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 356.85 | 359.82 | 360.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 14:15:00 | 353.60 | 351.10 | 353.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 14:15:00 | 353.60 | 351.10 | 353.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 353.60 | 351.10 | 353.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 353.60 | 351.10 | 353.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 350.75 | 351.03 | 353.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 351.20 | 351.03 | 353.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 352.15 | 351.26 | 353.22 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 357.85 | 354.53 | 354.19 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 351.20 | 354.00 | 354.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 350.30 | 352.83 | 353.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 348.40 | 347.85 | 349.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:00:00 | 348.40 | 347.85 | 349.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 350.95 | 346.17 | 347.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:45:00 | 350.80 | 346.17 | 347.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 351.50 | 347.23 | 347.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:30:00 | 351.80 | 347.23 | 347.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 352.30 | 348.25 | 348.10 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 341.75 | 346.95 | 347.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 337.00 | 345.09 | 346.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 10:15:00 | 347.00 | 345.48 | 346.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 10:15:00 | 347.00 | 345.48 | 346.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 347.00 | 345.48 | 346.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 11:00:00 | 347.00 | 345.48 | 346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 340.15 | 344.41 | 345.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 12:45:00 | 337.95 | 343.04 | 345.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 348.30 | 345.28 | 345.30 | SL hit (close>static) qty=1.00 sl=348.20 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 348.85 | 346.00 | 345.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 15:15:00 | 351.20 | 347.04 | 346.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 389.95 | 390.06 | 383.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 389.95 | 390.06 | 383.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 393.10 | 395.35 | 390.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:30:00 | 398.80 | 396.24 | 391.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 398.00 | 398.77 | 395.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 399.55 | 398.92 | 395.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:30:00 | 397.95 | 400.10 | 399.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 395.95 | 399.27 | 399.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:30:00 | 394.15 | 399.27 | 399.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 396.60 | 398.74 | 398.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 396.60 | 398.74 | 398.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 10:15:00 | 391.90 | 396.73 | 397.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 12:15:00 | 398.50 | 397.03 | 397.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 12:15:00 | 398.50 | 397.03 | 397.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 398.50 | 397.03 | 397.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 398.80 | 397.03 | 397.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 396.60 | 396.94 | 397.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 394.80 | 396.94 | 397.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 10:15:00 | 399.40 | 397.55 | 397.69 | SL hit (close>static) qty=1.00 sl=398.75 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 400.00 | 398.08 | 397.90 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 391.50 | 397.03 | 397.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 10:15:00 | 374.60 | 378.50 | 384.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 360.05 | 358.90 | 366.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:30:00 | 360.00 | 358.90 | 366.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 365.75 | 359.65 | 363.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 365.75 | 359.65 | 363.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 365.75 | 360.87 | 363.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:45:00 | 367.00 | 360.87 | 363.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 355.25 | 357.20 | 360.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 12:30:00 | 352.85 | 356.21 | 359.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 372.60 | 360.03 | 360.23 | SL hit (close>static) qty=1.00 sl=363.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 371.30 | 362.28 | 361.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 377.40 | 370.11 | 366.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 432.35 | 440.83 | 432.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 432.35 | 440.83 | 432.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 432.35 | 440.83 | 432.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 432.35 | 440.83 | 432.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 434.45 | 439.55 | 432.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:00:00 | 438.30 | 436.49 | 432.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 439.50 | 436.93 | 433.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 436.45 | 437.83 | 435.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 442.70 | 439.84 | 436.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 447.60 | 445.29 | 441.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:30:00 | 452.10 | 448.97 | 444.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 482.13 | 472.90 | 466.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 470.00 | 473.41 | 473.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 466.95 | 470.14 | 471.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 472.00 | 470.42 | 471.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 472.00 | 470.42 | 471.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 472.00 | 470.42 | 471.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 472.20 | 470.42 | 471.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 471.95 | 470.73 | 471.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:45:00 | 471.90 | 470.73 | 471.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 471.90 | 470.96 | 471.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 470.00 | 470.55 | 471.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 468.95 | 470.64 | 471.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 446.50 | 456.66 | 459.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 445.50 | 456.66 | 459.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 451.65 | 447.08 | 451.66 | SL hit (close>ema200) qty=0.50 sl=447.08 alert=retest2 |

### Cycle 137 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 432.10 | 430.79 | 430.65 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 429.85 | 430.58 | 430.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 428.00 | 430.07 | 430.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 12:15:00 | 419.95 | 419.94 | 422.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 419.95 | 419.94 | 422.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 422.85 | 420.30 | 422.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 422.85 | 420.30 | 422.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 421.00 | 420.44 | 422.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 419.00 | 420.44 | 422.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 415.30 | 419.41 | 421.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 413.70 | 418.73 | 420.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 425.50 | 421.87 | 421.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 425.50 | 421.87 | 421.76 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 418.55 | 421.21 | 421.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 415.80 | 418.59 | 419.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 419.40 | 418.50 | 419.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 13:15:00 | 419.40 | 418.50 | 419.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 419.40 | 418.50 | 419.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 419.40 | 418.50 | 419.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 418.55 | 418.51 | 419.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 418.55 | 418.51 | 419.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 418.95 | 418.60 | 419.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 419.35 | 418.60 | 419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 418.25 | 418.53 | 419.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 418.25 | 418.53 | 419.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 420.00 | 418.82 | 419.27 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 422.45 | 419.84 | 419.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 423.75 | 421.06 | 420.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 419.50 | 420.93 | 420.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 419.50 | 420.93 | 420.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 419.50 | 420.93 | 420.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:45:00 | 425.85 | 423.37 | 422.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:30:00 | 426.40 | 424.16 | 422.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 426.25 | 428.19 | 427.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 426.45 | 427.25 | 427.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 428.20 | 424.60 | 424.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 440.40 | 428.89 | 426.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 438.75 | 439.21 | 434.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 437.70 | 439.21 | 434.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 433.80 | 437.77 | 434.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 433.80 | 437.77 | 434.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 433.30 | 436.88 | 434.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 433.30 | 436.88 | 434.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 436.55 | 436.81 | 434.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 433.20 | 436.81 | 434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 434.45 | 436.34 | 434.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 434.45 | 436.34 | 434.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 433.75 | 435.82 | 434.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 437.20 | 435.82 | 434.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:30:00 | 434.95 | 436.08 | 434.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 434.90 | 434.66 | 434.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 431.00 | 433.93 | 434.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 431.00 | 433.93 | 434.17 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 438.30 | 435.03 | 434.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 440.15 | 436.05 | 435.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 437.75 | 439.22 | 437.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 437.75 | 439.22 | 437.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 437.75 | 439.22 | 437.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 437.75 | 439.22 | 437.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 438.45 | 439.06 | 437.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:15:00 | 438.25 | 439.06 | 437.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 440.85 | 439.42 | 437.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 442.25 | 439.92 | 438.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 442.60 | 441.98 | 439.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 435.10 | 439.97 | 440.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 435.10 | 439.97 | 440.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 433.35 | 438.64 | 439.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 425.75 | 423.78 | 426.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 425.75 | 423.78 | 426.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 425.75 | 423.78 | 426.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 425.70 | 423.78 | 426.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 426.45 | 424.32 | 426.18 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 438.30 | 427.69 | 427.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 442.55 | 432.71 | 429.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 440.60 | 443.49 | 439.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:15:00 | 441.35 | 443.49 | 439.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 439.50 | 442.69 | 439.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 439.50 | 442.69 | 439.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 438.40 | 441.84 | 439.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 435.75 | 441.84 | 439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 436.75 | 440.82 | 439.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 436.75 | 440.82 | 439.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 437.50 | 440.15 | 439.10 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 424.80 | 435.81 | 437.30 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 434.45 | 433.17 | 433.11 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 432.70 | 433.10 | 433.11 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 435.70 | 433.58 | 433.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 437.50 | 435.63 | 434.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 12:15:00 | 442.05 | 442.62 | 439.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:00:00 | 442.05 | 442.62 | 439.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 445.30 | 443.97 | 441.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 445.30 | 443.97 | 441.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 443.00 | 444.38 | 442.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 443.00 | 444.38 | 442.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 441.00 | 443.71 | 442.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 441.00 | 443.71 | 442.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 438.55 | 442.67 | 441.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 438.55 | 442.67 | 441.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 447.95 | 448.61 | 445.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:45:00 | 447.60 | 448.61 | 445.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 446.50 | 448.19 | 445.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 447.50 | 448.19 | 445.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 444.95 | 447.28 | 445.74 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 439.80 | 444.13 | 444.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 13:15:00 | 439.55 | 443.22 | 444.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 441.95 | 441.72 | 443.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 441.95 | 441.72 | 443.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 441.95 | 441.72 | 443.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 441.95 | 441.72 | 443.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 442.55 | 439.19 | 440.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 439.55 | 440.05 | 440.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 438.00 | 440.13 | 440.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 442.50 | 441.31 | 441.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 442.50 | 441.31 | 441.15 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 438.30 | 440.56 | 440.83 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 442.95 | 441.04 | 441.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 14:15:00 | 444.95 | 441.82 | 441.38 | Break + close above crossover candle high |

### Cycle 156 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 437.55 | 441.06 | 441.12 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 444.00 | 441.40 | 441.19 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 433.50 | 439.92 | 440.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 429.50 | 437.83 | 439.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 435.65 | 435.35 | 437.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 437.35 | 435.75 | 437.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 437.35 | 435.75 | 437.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 437.35 | 435.75 | 437.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 436.60 | 435.92 | 437.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 435.75 | 436.63 | 437.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 439.40 | 437.18 | 437.54 | SL hit (close>static) qty=1.00 sl=439.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 441.15 | 437.98 | 437.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 15:15:00 | 450.00 | 440.38 | 438.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 14:15:00 | 446.55 | 446.98 | 443.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 15:00:00 | 446.55 | 446.98 | 443.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 446.00 | 446.78 | 443.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 451.25 | 447.31 | 444.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 451.85 | 449.07 | 446.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 439.45 | 445.62 | 445.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 439.45 | 445.62 | 445.72 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 447.65 | 445.55 | 445.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 452.15 | 447.90 | 446.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 451.00 | 453.30 | 450.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 451.00 | 453.30 | 450.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 447.15 | 452.07 | 449.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 447.50 | 452.07 | 449.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 445.00 | 450.66 | 449.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 445.00 | 450.66 | 449.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 446.00 | 449.72 | 449.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 448.25 | 449.10 | 448.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 444.85 | 448.07 | 448.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 444.85 | 448.07 | 448.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 440.10 | 445.70 | 447.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 451.60 | 446.72 | 447.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 451.60 | 446.72 | 447.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 451.60 | 446.72 | 447.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 452.70 | 446.72 | 447.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 451.75 | 447.72 | 447.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 451.75 | 447.72 | 447.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 452.55 | 448.69 | 448.26 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 443.65 | 448.02 | 448.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 11:15:00 | 441.50 | 446.71 | 447.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 15:15:00 | 441.00 | 440.80 | 443.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:15:00 | 440.80 | 440.80 | 443.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 440.80 | 440.80 | 442.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 438.35 | 440.31 | 442.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 443.50 | 442.23 | 442.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 443.50 | 442.23 | 442.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 445.40 | 442.78 | 442.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 442.10 | 442.64 | 442.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 442.10 | 442.64 | 442.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 442.10 | 442.64 | 442.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 442.10 | 442.64 | 442.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 441.65 | 442.45 | 442.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 445.70 | 442.53 | 442.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 447.00 | 442.53 | 442.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 450.15 | 453.19 | 453.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 450.15 | 453.19 | 453.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 445.35 | 451.36 | 452.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 450.95 | 444.07 | 446.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 450.95 | 444.07 | 446.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 450.95 | 444.07 | 446.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 451.90 | 444.07 | 446.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 448.80 | 445.02 | 447.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:00:00 | 446.20 | 445.25 | 446.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 445.50 | 445.28 | 446.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 453.00 | 447.52 | 447.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 453.00 | 447.52 | 447.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 456.50 | 451.45 | 449.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 456.60 | 457.49 | 455.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:15:00 | 459.05 | 457.49 | 455.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 458.00 | 457.59 | 455.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 455.45 | 457.59 | 455.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 13:15:00 | 482.00 | 476.14 | 469.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 477.35 | 479.64 | 474.87 | SL hit (close<ema200) qty=0.50 sl=479.64 alert=retest1 |

### Cycle 168 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 473.55 | 476.18 | 476.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 467.60 | 473.86 | 475.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 477.40 | 471.34 | 472.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 477.40 | 471.34 | 472.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 477.40 | 471.34 | 472.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 477.40 | 471.34 | 472.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 484.90 | 474.05 | 473.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 488.85 | 481.21 | 477.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 496.70 | 497.10 | 490.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:00:00 | 496.70 | 497.10 | 490.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 494.55 | 495.23 | 491.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 493.20 | 495.23 | 491.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 493.00 | 496.07 | 493.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 493.00 | 496.07 | 493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 493.10 | 495.95 | 494.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 493.50 | 495.95 | 494.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 494.55 | 495.67 | 494.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 494.00 | 495.67 | 494.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 493.85 | 495.31 | 494.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 493.85 | 495.31 | 494.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 495.00 | 495.24 | 494.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 492.60 | 495.24 | 494.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 491.35 | 494.47 | 493.99 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 488.20 | 492.66 | 493.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 487.80 | 491.69 | 492.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 490.05 | 490.04 | 491.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 488.75 | 490.04 | 491.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 485.30 | 489.09 | 490.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 486.70 | 489.09 | 490.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 487.10 | 487.29 | 489.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 489.55 | 487.29 | 489.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 492.55 | 488.34 | 489.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 489.25 | 488.34 | 489.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 494.85 | 489.64 | 489.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 494.85 | 489.64 | 489.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 496.50 | 491.01 | 490.57 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 465.55 | 487.76 | 489.55 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 495.85 | 483.03 | 482.92 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 486.65 | 488.34 | 488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 480.45 | 486.76 | 487.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 484.35 | 484.01 | 485.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:30:00 | 484.05 | 484.01 | 485.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 487.90 | 484.79 | 486.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 487.90 | 484.79 | 486.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 483.00 | 484.43 | 485.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 476.90 | 484.43 | 485.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 490.70 | 485.03 | 485.18 | SL hit (close>static) qty=1.00 sl=488.15 alert=retest2 |

### Cycle 175 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 490.35 | 486.10 | 485.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 10:15:00 | 497.30 | 489.27 | 487.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 490.90 | 492.34 | 489.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:45:00 | 490.45 | 492.34 | 489.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 490.00 | 491.54 | 489.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 493.40 | 490.54 | 489.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 492.85 | 491.62 | 490.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 511.10 | 514.35 | 514.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 511.10 | 514.35 | 514.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 508.45 | 513.17 | 514.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 12:15:00 | 509.60 | 509.36 | 511.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:00:00 | 509.60 | 509.36 | 511.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 498.00 | 490.66 | 495.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 498.00 | 490.66 | 495.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 496.95 | 491.92 | 495.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 487.00 | 491.92 | 495.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:30:00 | 486.20 | 491.55 | 493.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:30:00 | 489.90 | 490.80 | 491.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 491.10 | 490.88 | 491.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 491.00 | 490.51 | 491.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 491.00 | 490.51 | 491.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 491.00 | 490.60 | 491.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 484.60 | 490.60 | 491.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 487.25 | 489.93 | 491.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 482.00 | 487.18 | 488.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 482.65 | 484.85 | 487.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:00:00 | 481.75 | 481.71 | 484.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:15:00 | 466.55 | 470.96 | 475.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 12:15:00 | 465.40 | 469.83 | 474.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 462.65 | 466.68 | 471.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:15:00 | 461.89 | 465.29 | 470.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 12:15:00 | 457.90 | 463.93 | 469.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 12:15:00 | 458.52 | 463.93 | 469.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 12:15:00 | 457.66 | 463.93 | 469.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |

### Cycle 177 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 471.50 | 466.23 | 465.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 472.10 | 468.64 | 467.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 469.95 | 470.15 | 468.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 469.95 | 470.15 | 468.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 469.95 | 470.15 | 468.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 468.60 | 470.15 | 468.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 466.05 | 469.33 | 468.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 468.15 | 469.33 | 468.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 467.25 | 468.91 | 468.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 466.05 | 468.91 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 466.50 | 468.43 | 468.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 466.90 | 468.43 | 468.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 468.15 | 468.53 | 468.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 468.15 | 468.53 | 468.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 466.60 | 468.14 | 468.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 466.60 | 468.14 | 468.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 464.95 | 467.50 | 467.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 463.90 | 466.00 | 466.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 465.00 | 464.99 | 466.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 15:00:00 | 465.00 | 464.99 | 466.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 469.15 | 465.74 | 466.23 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 470.20 | 466.63 | 466.59 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 464.60 | 466.32 | 466.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 462.20 | 465.50 | 466.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 462.05 | 461.73 | 463.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 11:00:00 | 462.05 | 461.73 | 463.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 460.55 | 461.30 | 462.57 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 464.80 | 462.25 | 462.01 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 460.75 | 461.88 | 461.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 459.45 | 461.39 | 461.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 461.70 | 461.40 | 461.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 15:15:00 | 461.70 | 461.40 | 461.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 461.70 | 461.40 | 461.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 459.10 | 461.40 | 461.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 459.50 | 461.02 | 461.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:30:00 | 456.50 | 458.79 | 459.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 449.45 | 446.88 | 446.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 449.45 | 446.88 | 446.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 451.35 | 447.92 | 447.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 461.00 | 463.44 | 458.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 12:30:00 | 460.40 | 463.44 | 458.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 459.35 | 462.62 | 458.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 459.60 | 462.62 | 458.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 459.55 | 462.01 | 459.01 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 452.50 | 457.94 | 458.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 451.75 | 456.70 | 457.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 457.25 | 448.77 | 451.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 457.25 | 448.77 | 451.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 457.25 | 448.77 | 451.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 443.90 | 449.25 | 450.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 455.00 | 452.02 | 451.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 455.00 | 452.02 | 451.62 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 447.05 | 450.66 | 451.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 443.50 | 449.38 | 450.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 452.05 | 446.05 | 447.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 11:15:00 | 452.05 | 446.05 | 447.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 452.05 | 446.05 | 447.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 452.05 | 446.05 | 447.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 451.80 | 447.20 | 447.56 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 450.50 | 447.86 | 447.83 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 447.00 | 447.77 | 447.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 444.55 | 446.99 | 447.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 13:15:00 | 441.15 | 440.87 | 442.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 441.15 | 440.87 | 442.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 439.55 | 439.82 | 441.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:15:00 | 441.95 | 439.82 | 441.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 441.30 | 440.12 | 441.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:15:00 | 438.90 | 440.12 | 441.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 416.95 | 428.44 | 431.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 430.30 | 428.82 | 431.53 | SL hit (close>ema200) qty=0.50 sl=428.82 alert=retest2 |

### Cycle 189 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 438.40 | 432.03 | 431.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 455.10 | 440.17 | 435.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 468.55 | 469.78 | 461.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 468.55 | 469.78 | 461.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 470.05 | 469.83 | 461.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 466.50 | 469.83 | 461.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 465.50 | 470.74 | 466.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 465.50 | 470.74 | 466.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 468.55 | 470.30 | 467.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 472.40 | 469.85 | 467.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:30:00 | 471.55 | 471.72 | 468.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 475.40 | 471.58 | 469.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 462.60 | 468.65 | 469.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 462.60 | 468.65 | 469.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 458.55 | 462.53 | 465.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 464.00 | 461.01 | 463.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 464.00 | 461.01 | 463.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 464.00 | 461.01 | 463.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 462.20 | 461.01 | 463.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 461.90 | 461.18 | 463.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 12:45:00 | 460.25 | 461.38 | 462.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 460.90 | 461.77 | 462.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 468.50 | 463.39 | 463.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 468.50 | 463.39 | 463.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 473.80 | 469.51 | 467.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 15:15:00 | 478.05 | 478.76 | 475.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:15:00 | 478.25 | 478.76 | 475.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 473.45 | 477.69 | 475.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 473.45 | 477.69 | 475.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 474.60 | 477.08 | 475.21 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 467.15 | 473.66 | 474.08 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 476.60 | 472.68 | 472.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 479.00 | 474.18 | 473.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 13:15:00 | 474.10 | 476.97 | 475.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 13:15:00 | 474.10 | 476.97 | 475.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 474.10 | 476.97 | 475.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 474.10 | 476.97 | 475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 475.00 | 476.58 | 475.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 473.50 | 476.58 | 475.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 471.70 | 475.60 | 475.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 469.70 | 475.60 | 475.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 469.00 | 474.28 | 474.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 466.90 | 472.81 | 473.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 459.30 | 457.32 | 461.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:30:00 | 458.05 | 457.32 | 461.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 458.55 | 458.24 | 460.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:45:00 | 457.55 | 458.34 | 460.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 454.75 | 458.37 | 460.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 467.50 | 458.13 | 458.68 | SL hit (close>static) qty=1.00 sl=460.95 alert=retest2 |

### Cycle 195 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 469.15 | 460.33 | 459.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 475.50 | 467.73 | 464.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 471.60 | 471.97 | 468.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:30:00 | 469.50 | 471.97 | 468.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 467.90 | 470.65 | 468.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 467.35 | 470.65 | 468.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 468.45 | 470.21 | 468.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 467.80 | 470.21 | 468.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 468.80 | 469.93 | 468.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 468.80 | 469.93 | 468.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 465.55 | 469.05 | 468.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 465.55 | 469.05 | 468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 464.80 | 468.20 | 467.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 461.75 | 468.20 | 467.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 460.60 | 466.68 | 467.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 445.60 | 456.04 | 457.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 453.80 | 449.07 | 452.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 453.80 | 449.07 | 452.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 453.80 | 449.07 | 452.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 453.70 | 449.07 | 452.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 454.10 | 450.08 | 452.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:30:00 | 456.00 | 450.08 | 452.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 448.50 | 451.31 | 452.38 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 455.55 | 453.40 | 453.13 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 444.65 | 452.58 | 452.96 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 459.10 | 453.55 | 453.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 463.00 | 455.44 | 454.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 460.00 | 460.15 | 457.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:00:00 | 459.45 | 460.01 | 458.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 455.70 | 459.15 | 457.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 454.70 | 459.15 | 457.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 456.15 | 458.55 | 457.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:30:00 | 460.00 | 457.76 | 457.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 459.20 | 457.61 | 457.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 463.15 | 458.69 | 458.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 454.20 | 459.64 | 460.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 454.20 | 459.64 | 460.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 15:15:00 | 450.60 | 457.83 | 459.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 459.65 | 457.90 | 459.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 459.65 | 457.90 | 459.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 459.65 | 457.90 | 459.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 460.40 | 457.90 | 459.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 456.60 | 457.64 | 458.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 458.20 | 457.64 | 458.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 456.70 | 457.45 | 458.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 458.65 | 457.45 | 458.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 456.90 | 457.34 | 458.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 455.40 | 457.34 | 458.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 453.40 | 456.55 | 457.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 455.50 | 451.40 | 451.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 455.50 | 451.40 | 451.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 462.30 | 453.58 | 452.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 455.60 | 457.61 | 455.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 455.60 | 457.61 | 455.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 455.60 | 457.61 | 455.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 10:15:00 | 458.50 | 457.61 | 455.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 450.05 | 456.75 | 456.33 | SL hit (close<static) qty=1.00 sl=451.80 alert=retest2 |

### Cycle 202 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 450.60 | 455.52 | 455.81 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 464.85 | 456.09 | 455.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 469.00 | 460.09 | 457.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 458.70 | 461.69 | 458.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 458.70 | 461.69 | 458.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 458.70 | 461.69 | 458.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 458.70 | 461.69 | 458.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 454.90 | 460.33 | 458.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 454.90 | 460.33 | 458.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 454.90 | 459.25 | 458.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:00:00 | 454.90 | 459.25 | 458.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 452.05 | 461.55 | 460.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 452.05 | 461.55 | 460.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 453.35 | 459.91 | 459.46 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 11:15:00 | 454.00 | 458.73 | 458.97 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 469.00 | 458.97 | 458.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 473.15 | 462.57 | 460.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 10:15:00 | 467.35 | 468.32 | 465.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:00:00 | 467.35 | 468.32 | 465.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 464.85 | 467.42 | 465.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 464.85 | 467.42 | 465.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 464.85 | 466.90 | 465.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 463.05 | 466.90 | 465.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 464.70 | 466.14 | 465.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 471.60 | 466.14 | 465.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 465.00 | 468.95 | 467.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 465.15 | 468.40 | 467.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 466.00 | 467.92 | 467.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 470.25 | 468.13 | 467.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:30:00 | 466.30 | 468.13 | 467.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2026-04-21 09:15:00 | 511.50 | 505.78 | 499.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 521.55 | 525.40 | 525.56 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 11:15:00 | 520.85 | 519.39 | 519.33 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 518.80 | 519.27 | 519.28 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 530.35 | 521.49 | 520.28 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 11:45:00 | 556.60 | 2024-04-16 12:15:00 | 560.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-04-12 12:15:00 | 556.40 | 2024-04-16 12:15:00 | 560.15 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-04-29 10:15:00 | 560.05 | 2024-04-29 12:15:00 | 567.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-05-06 13:00:00 | 558.25 | 2024-05-10 09:15:00 | 530.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 13:00:00 | 558.25 | 2024-05-10 11:15:00 | 537.85 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2024-05-29 10:30:00 | 521.60 | 2024-06-04 12:15:00 | 495.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 09:15:00 | 519.10 | 2024-06-04 12:15:00 | 493.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 09:30:00 | 521.40 | 2024-06-04 12:15:00 | 495.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 10:30:00 | 521.60 | 2024-06-06 09:15:00 | 503.55 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2024-05-31 09:15:00 | 519.10 | 2024-06-06 09:15:00 | 503.55 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2024-06-03 09:30:00 | 521.40 | 2024-06-06 09:15:00 | 503.55 | STOP_HIT | 0.50 | 3.42% |
| BUY | retest2 | 2024-06-14 09:15:00 | 518.00 | 2024-06-21 10:15:00 | 519.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-07-04 10:30:00 | 586.75 | 2024-07-09 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-07-04 11:15:00 | 586.20 | 2024-07-09 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-07-04 11:45:00 | 589.00 | 2024-07-09 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-07-19 09:15:00 | 591.60 | 2024-07-19 15:15:00 | 584.55 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-07-19 10:00:00 | 590.00 | 2024-07-19 15:15:00 | 584.55 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-07-23 12:15:00 | 574.10 | 2024-07-23 14:15:00 | 595.00 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-07-23 13:00:00 | 581.25 | 2024-07-23 14:15:00 | 595.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-07-23 13:30:00 | 583.05 | 2024-07-23 14:15:00 | 595.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-07-29 09:15:00 | 620.95 | 2024-07-29 10:15:00 | 610.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-07-29 12:00:00 | 615.85 | 2024-07-30 09:15:00 | 609.15 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-29 12:30:00 | 616.00 | 2024-07-30 09:15:00 | 609.15 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-07-30 11:30:00 | 616.10 | 2024-07-31 09:15:00 | 607.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-01 11:30:00 | 608.10 | 2024-08-05 09:15:00 | 577.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 11:30:00 | 608.10 | 2024-08-07 12:15:00 | 577.95 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2024-08-26 09:15:00 | 608.25 | 2024-08-29 12:15:00 | 602.95 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-08-26 12:30:00 | 604.00 | 2024-08-29 12:15:00 | 602.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-08-26 13:30:00 | 605.00 | 2024-08-29 12:15:00 | 602.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-09-02 15:15:00 | 622.00 | 2024-09-09 13:15:00 | 629.80 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2024-09-12 10:45:00 | 615.45 | 2024-09-20 13:15:00 | 613.20 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-09-13 09:45:00 | 616.05 | 2024-09-20 13:15:00 | 613.20 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2024-09-13 12:00:00 | 617.00 | 2024-09-20 13:15:00 | 613.20 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-09-17 10:45:00 | 617.40 | 2024-09-20 13:15:00 | 613.20 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2024-10-10 09:15:00 | 563.90 | 2024-10-15 14:15:00 | 562.15 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest1 | 2024-10-18 09:15:00 | 540.50 | 2024-10-21 09:15:00 | 556.55 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-10-21 11:00:00 | 550.00 | 2024-10-23 12:15:00 | 549.75 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-10-22 09:30:00 | 549.55 | 2024-10-23 12:15:00 | 549.75 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-10-22 10:15:00 | 548.00 | 2024-10-23 12:15:00 | 549.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-10-23 12:15:00 | 545.75 | 2024-10-23 12:15:00 | 549.75 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-10-24 10:30:00 | 544.60 | 2024-10-30 09:15:00 | 517.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 14:15:00 | 545.40 | 2024-10-30 09:15:00 | 518.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 15:00:00 | 545.55 | 2024-10-30 09:15:00 | 518.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:30:00 | 537.30 | 2024-10-30 09:15:00 | 510.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-28 09:15:00 | 534.15 | 2024-10-30 09:15:00 | 507.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-28 12:15:00 | 538.20 | 2024-10-30 09:15:00 | 511.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-28 13:15:00 | 538.30 | 2024-10-30 09:15:00 | 511.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-29 09:15:00 | 538.20 | 2024-10-30 09:15:00 | 511.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-29 13:30:00 | 531.05 | 2024-10-30 09:15:00 | 504.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-29 15:00:00 | 535.15 | 2024-10-30 09:15:00 | 508.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 10:30:00 | 544.60 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 5.63% |
| SELL | retest2 | 2024-10-24 14:15:00 | 545.40 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2024-10-24 15:00:00 | 545.55 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2024-10-25 09:30:00 | 537.30 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2024-10-28 09:15:00 | 534.15 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2024-10-28 12:15:00 | 538.20 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 4.51% |
| SELL | retest2 | 2024-10-28 13:15:00 | 538.30 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-10-29 09:15:00 | 538.20 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 4.51% |
| SELL | retest2 | 2024-10-29 13:30:00 | 531.05 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2024-10-29 15:00:00 | 535.15 | 2024-11-01 17:15:00 | 513.95 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2024-11-12 09:30:00 | 470.95 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2024-11-12 10:00:00 | 470.70 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2024-11-12 13:30:00 | 471.05 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2024-11-13 09:15:00 | 471.05 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2024-11-18 09:15:00 | 465.45 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-11-18 11:45:00 | 465.30 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-11-18 12:45:00 | 466.40 | 2024-11-25 11:15:00 | 464.40 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2024-12-13 10:00:00 | 462.05 | 2024-12-16 09:15:00 | 467.85 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-12-13 11:15:00 | 462.70 | 2024-12-16 09:15:00 | 467.85 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-12-13 12:30:00 | 462.20 | 2024-12-16 09:15:00 | 467.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-12-13 13:15:00 | 462.50 | 2024-12-16 09:15:00 | 467.85 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-12-19 11:30:00 | 484.70 | 2024-12-23 09:15:00 | 475.65 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-12-19 12:15:00 | 484.30 | 2024-12-23 09:15:00 | 475.65 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-12-31 09:15:00 | 477.10 | 2025-01-01 09:15:00 | 485.75 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-12-31 12:00:00 | 478.00 | 2025-01-01 09:15:00 | 485.75 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-12-31 12:30:00 | 476.95 | 2025-01-01 09:15:00 | 485.75 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-12-31 13:15:00 | 476.15 | 2025-01-01 09:15:00 | 485.75 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-31 15:15:00 | 474.85 | 2025-01-01 09:15:00 | 485.75 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-01-08 09:15:00 | 474.20 | 2025-01-16 12:15:00 | 468.40 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2025-01-22 09:15:00 | 458.20 | 2025-01-23 13:15:00 | 463.75 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-01-22 10:15:00 | 460.45 | 2025-01-23 13:15:00 | 463.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-01-22 10:45:00 | 460.00 | 2025-01-23 13:15:00 | 463.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-23 09:15:00 | 457.05 | 2025-01-23 13:15:00 | 463.75 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-01-23 12:00:00 | 462.45 | 2025-01-23 13:15:00 | 463.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-01-23 13:00:00 | 462.10 | 2025-01-23 13:15:00 | 463.75 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-02-10 09:15:00 | 428.90 | 2025-02-11 11:15:00 | 407.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 428.90 | 2025-02-13 09:15:00 | 403.20 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2025-03-12 10:45:00 | 356.80 | 2025-03-19 09:15:00 | 359.35 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-03-12 11:15:00 | 355.25 | 2025-03-19 09:15:00 | 359.35 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-03-13 11:00:00 | 357.65 | 2025-03-19 09:15:00 | 359.35 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-03-13 13:00:00 | 356.70 | 2025-03-20 10:15:00 | 360.25 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-03-18 12:00:00 | 354.80 | 2025-03-20 10:15:00 | 360.25 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-03-18 12:30:00 | 354.50 | 2025-03-20 10:15:00 | 360.25 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-03-18 15:00:00 | 354.55 | 2025-03-20 10:15:00 | 360.25 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-03-19 11:15:00 | 354.90 | 2025-03-20 10:15:00 | 360.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-04-07 12:45:00 | 337.95 | 2025-04-08 13:15:00 | 348.30 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-04-22 11:30:00 | 398.80 | 2025-04-25 12:15:00 | 396.60 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-04-23 10:15:00 | 398.00 | 2025-04-25 12:15:00 | 396.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-04-23 11:00:00 | 399.55 | 2025-04-25 12:15:00 | 396.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-04-25 10:30:00 | 397.95 | 2025-04-25 12:15:00 | 396.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-04-28 14:15:00 | 394.80 | 2025-04-29 10:15:00 | 399.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-05-09 12:30:00 | 352.85 | 2025-05-12 09:15:00 | 372.60 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2025-05-20 15:00:00 | 438.30 | 2025-05-29 09:15:00 | 482.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 09:45:00 | 439.50 | 2025-05-29 09:15:00 | 483.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 15:00:00 | 436.45 | 2025-05-29 09:15:00 | 480.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:45:00 | 442.70 | 2025-06-03 13:15:00 | 470.00 | STOP_HIT | 1.00 | 6.17% |
| BUY | retest2 | 2025-05-23 12:30:00 | 452.10 | 2025-06-03 13:15:00 | 470.00 | STOP_HIT | 1.00 | 3.96% |
| SELL | retest2 | 2025-06-05 13:30:00 | 470.00 | 2025-06-13 09:15:00 | 446.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 15:15:00 | 468.95 | 2025-06-13 09:15:00 | 445.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 13:30:00 | 470.00 | 2025-06-16 11:15:00 | 451.65 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-06-05 15:15:00 | 468.95 | 2025-06-16 11:15:00 | 451.65 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-07-01 10:45:00 | 413.70 | 2025-07-01 15:15:00 | 425.50 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-07-08 13:45:00 | 425.85 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-08 14:30:00 | 426.40 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-10 14:15:00 | 426.25 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-11 09:45:00 | 426.45 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-18 09:15:00 | 437.20 | 2025-07-18 15:15:00 | 431.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-18 10:30:00 | 434.95 | 2025-07-18 15:15:00 | 431.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-18 14:30:00 | 434.90 | 2025-07-18 15:15:00 | 431.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-22 13:45:00 | 442.25 | 2025-07-24 10:15:00 | 435.10 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-23 11:15:00 | 442.60 | 2025-07-24 10:15:00 | 435.10 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-21 12:15:00 | 439.55 | 2025-08-22 10:15:00 | 442.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-08-22 09:15:00 | 438.00 | 2025-08-22 10:15:00 | 442.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-28 12:30:00 | 435.75 | 2025-08-28 13:15:00 | 439.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-01 09:30:00 | 451.25 | 2025-09-02 13:15:00 | 439.45 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-09-02 09:30:00 | 451.85 | 2025-09-02 13:15:00 | 439.45 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-09-05 09:45:00 | 448.25 | 2025-09-05 11:15:00 | 444.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-11 12:45:00 | 438.35 | 2025-09-15 14:15:00 | 443.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-16 13:45:00 | 445.70 | 2025-09-26 10:15:00 | 450.15 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-09-16 14:15:00 | 447.00 | 2025-09-26 10:15:00 | 450.15 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-09-30 12:00:00 | 446.20 | 2025-10-01 09:15:00 | 453.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-30 12:45:00 | 445.50 | 2025-10-01 09:15:00 | 453.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2025-10-07 09:15:00 | 459.05 | 2025-10-08 13:15:00 | 482.00 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-07 09:15:00 | 459.05 | 2025-10-09 13:15:00 | 477.35 | STOP_HIT | 0.50 | 3.99% |
| BUY | retest2 | 2025-10-10 12:00:00 | 478.65 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-10 13:30:00 | 477.85 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-10 14:45:00 | 479.90 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-13 12:00:00 | 480.45 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-07 09:15:00 | 476.90 | 2025-11-07 14:15:00 | 490.70 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-11-12 10:45:00 | 493.40 | 2025-11-19 13:15:00 | 511.10 | STOP_HIT | 1.00 | 3.59% |
| BUY | retest2 | 2025-11-12 12:45:00 | 492.85 | 2025-11-19 13:15:00 | 511.10 | STOP_HIT | 1.00 | 3.70% |
| SELL | retest2 | 2025-11-25 09:15:00 | 487.00 | 2025-12-04 11:15:00 | 466.55 | PARTIAL | 0.50 | 4.20% |
| SELL | retest2 | 2025-11-26 10:30:00 | 486.20 | 2025-12-04 12:15:00 | 465.40 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-11-27 10:30:00 | 489.90 | 2025-12-05 10:15:00 | 462.65 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2025-11-27 11:45:00 | 491.10 | 2025-12-05 11:15:00 | 461.89 | PARTIAL | 0.50 | 5.95% |
| SELL | retest2 | 2025-12-01 11:45:00 | 482.00 | 2025-12-05 12:15:00 | 457.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 482.65 | 2025-12-05 12:15:00 | 458.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 14:00:00 | 481.75 | 2025-12-05 12:15:00 | 457.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 09:15:00 | 487.00 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2025-11-26 10:30:00 | 486.20 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-11-27 10:30:00 | 489.90 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-27 11:45:00 | 491.10 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2025-12-01 11:45:00 | 482.00 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-12-01 14:30:00 | 482.65 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-12-02 14:00:00 | 481.75 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-12-24 11:30:00 | 456.50 | 2025-12-31 10:15:00 | 449.45 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2026-01-09 09:15:00 | 443.90 | 2026-01-09 11:15:00 | 455.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-01-20 12:15:00 | 438.90 | 2026-01-27 09:15:00 | 416.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:15:00 | 438.90 | 2026-01-27 10:15:00 | 430.30 | STOP_HIT | 0.50 | 1.96% |
| BUY | retest2 | 2026-02-03 09:15:00 | 472.40 | 2026-02-05 09:15:00 | 462.60 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-02-03 11:30:00 | 471.55 | 2026-02-05 09:15:00 | 462.60 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-04 09:15:00 | 475.40 | 2026-02-05 09:15:00 | 462.60 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-09 12:45:00 | 460.25 | 2026-02-10 10:15:00 | 468.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-02-09 14:30:00 | 460.90 | 2026-02-10 10:15:00 | 468.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-23 14:45:00 | 457.55 | 2026-02-25 09:15:00 | 467.50 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-24 09:15:00 | 454.75 | 2026-02-25 09:15:00 | 467.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-03-16 14:30:00 | 460.00 | 2026-03-19 14:15:00 | 454.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-03-17 09:15:00 | 459.20 | 2026-03-19 14:15:00 | 454.20 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-03-18 09:15:00 | 463.15 | 2026-03-19 14:15:00 | 454.20 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-03-20 14:15:00 | 455.40 | 2026-03-24 15:15:00 | 455.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-03-20 15:00:00 | 453.40 | 2026-03-24 15:15:00 | 455.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-03-27 10:15:00 | 458.50 | 2026-03-30 09:15:00 | 450.05 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-04-10 09:15:00 | 471.60 | 2026-04-21 09:15:00 | 511.50 | TARGET_HIT | 1.00 | 8.46% |
| BUY | retest2 | 2026-04-13 09:45:00 | 465.00 | 2026-04-21 09:15:00 | 511.67 | TARGET_HIT | 1.00 | 10.04% |
| BUY | retest2 | 2026-04-13 12:00:00 | 465.15 | 2026-04-21 10:15:00 | 512.60 | TARGET_HIT | 1.00 | 10.20% |
| BUY | retest2 | 2026-04-13 13:00:00 | 466.00 | 2026-04-24 09:15:00 | 518.76 | TARGET_HIT | 1.00 | 11.32% |
| BUY | retest2 | 2026-04-23 09:30:00 | 511.10 | 2026-04-29 09:15:00 | 562.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 09:15:00 | 515.25 | 2026-04-29 09:15:00 | 566.78 | TARGET_HIT | 1.00 | 10.00% |
