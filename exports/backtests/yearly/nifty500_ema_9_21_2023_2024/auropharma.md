# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1487.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 239 |
| ALERT1 | 151 |
| ALERT2 | 150 |
| ALERT2_SKIP | 77 |
| ALERT3 | 436 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 193 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 193 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 204 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 142
- **Target hits / Stop hits / Partials:** 2 / 193 / 9
- **Avg / median % per leg:** -0.27% / -0.91%
- **Sum % (uncompounded):** -54.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 108 | 28 | 25.9% | 1 | 107 | 0 | -0.51% | -54.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 108 | 28 | 25.9% | 1 | 107 | 0 | -0.51% | -54.9% |
| SELL (all) | 96 | 34 | 35.4% | 1 | 86 | 9 | 0.01% | 0.8% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.93% | 15.7% |
| SELL @ 3rd Alert (retest2) | 92 | 30 | 32.6% | 1 | 84 | 7 | -0.16% | -14.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.93% | 15.7% |
| retest2 (combined) | 200 | 58 | 29.0% | 2 | 191 | 7 | -0.35% | -69.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 11:15:00 | 614.00 | 610.41 | 610.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 638.10 | 617.12 | 613.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 09:15:00 | 633.05 | 635.39 | 626.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-17 10:00:00 | 633.05 | 635.39 | 626.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 627.40 | 633.26 | 627.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 11:45:00 | 626.75 | 633.26 | 627.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 625.00 | 631.61 | 627.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 12:45:00 | 624.75 | 631.61 | 627.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 620.70 | 629.43 | 626.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 14:00:00 | 620.70 | 629.43 | 626.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 09:15:00 | 611.10 | 622.74 | 623.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 12:15:00 | 610.75 | 617.55 | 621.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 604.90 | 594.87 | 602.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 604.90 | 594.87 | 602.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 604.90 | 594.87 | 602.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 604.30 | 594.87 | 602.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 603.50 | 596.60 | 602.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 11:45:00 | 601.05 | 597.45 | 602.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:30:00 | 601.15 | 597.98 | 602.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 15:15:00 | 607.00 | 600.57 | 602.45 | SL hit (close>static) qty=1.00 sl=605.80 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 609.40 | 604.21 | 603.89 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 10:15:00 | 600.40 | 603.96 | 604.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 11:15:00 | 598.95 | 602.32 | 603.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 605.05 | 601.14 | 602.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 605.05 | 601.14 | 602.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 605.05 | 601.14 | 602.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 605.05 | 601.14 | 602.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 606.70 | 602.25 | 602.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 09:15:00 | 598.45 | 602.25 | 602.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 10:30:00 | 599.00 | 602.15 | 602.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 13:15:00 | 604.15 | 602.93 | 602.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 604.15 | 602.93 | 602.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 14:15:00 | 614.05 | 605.15 | 603.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 09:15:00 | 587.70 | 603.42 | 603.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 587.70 | 603.42 | 603.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 587.70 | 603.42 | 603.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:30:00 | 590.35 | 603.42 | 603.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 10:15:00 | 599.45 | 602.63 | 603.04 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 11:15:00 | 608.35 | 603.77 | 603.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 14:15:00 | 610.50 | 606.19 | 604.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 09:15:00 | 647.35 | 652.65 | 641.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-01 10:00:00 | 647.35 | 652.65 | 641.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 655.75 | 659.43 | 657.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:45:00 | 655.75 | 659.43 | 657.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 657.00 | 658.95 | 657.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:30:00 | 656.00 | 658.95 | 657.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 12:15:00 | 654.15 | 657.14 | 656.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 14:00:00 | 655.45 | 656.80 | 656.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 12:15:00 | 665.45 | 666.52 | 666.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 12:15:00 | 665.45 | 666.52 | 666.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 13:15:00 | 663.65 | 665.95 | 666.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 665.75 | 665.36 | 665.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 665.75 | 665.36 | 665.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 665.75 | 665.36 | 665.91 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 670.40 | 666.39 | 666.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 12:15:00 | 672.80 | 667.67 | 666.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 10:15:00 | 672.35 | 673.67 | 670.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 11:00:00 | 672.35 | 673.67 | 670.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 675.05 | 673.95 | 671.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 12:45:00 | 676.50 | 674.21 | 671.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 14:15:00 | 669.65 | 672.89 | 671.32 | SL hit (close<static) qty=1.00 sl=670.55 alert=retest2 |

### Cycle 10 — SELL (started 2023-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 12:15:00 | 678.20 | 680.57 | 680.61 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 681.80 | 680.67 | 680.64 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 15:15:00 | 679.15 | 680.37 | 680.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 10:15:00 | 678.10 | 679.72 | 680.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 12:15:00 | 681.10 | 679.66 | 680.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 12:15:00 | 681.10 | 679.66 | 680.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 12:15:00 | 681.10 | 679.66 | 680.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 13:00:00 | 681.10 | 679.66 | 680.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 13:15:00 | 682.10 | 680.14 | 680.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 13:30:00 | 683.95 | 680.14 | 680.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 14:15:00 | 682.40 | 680.60 | 680.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 683.60 | 681.41 | 680.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 10:15:00 | 678.90 | 680.91 | 680.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 678.90 | 680.91 | 680.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 678.90 | 680.91 | 680.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 678.90 | 680.91 | 680.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 675.80 | 679.89 | 680.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 13:15:00 | 669.40 | 676.89 | 678.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 09:15:00 | 675.10 | 674.94 | 677.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 09:45:00 | 673.20 | 674.94 | 677.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 680.45 | 676.04 | 677.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:30:00 | 684.20 | 676.04 | 677.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 684.95 | 677.82 | 678.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 11:30:00 | 687.75 | 677.82 | 678.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 12:15:00 | 691.55 | 680.57 | 679.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 13:15:00 | 697.00 | 683.85 | 681.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 10:15:00 | 727.95 | 728.30 | 721.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 10:45:00 | 727.50 | 728.30 | 721.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 721.70 | 726.77 | 723.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:45:00 | 718.00 | 726.77 | 723.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 717.70 | 724.95 | 723.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:00:00 | 717.70 | 724.95 | 723.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 12:15:00 | 715.30 | 721.25 | 721.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 13:15:00 | 712.70 | 719.54 | 720.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 10:15:00 | 716.50 | 716.05 | 718.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-04 11:00:00 | 716.50 | 716.05 | 718.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 721.60 | 715.65 | 716.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:45:00 | 723.80 | 715.65 | 716.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 720.50 | 716.62 | 717.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 11:15:00 | 723.50 | 716.62 | 717.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 11:15:00 | 725.35 | 718.37 | 718.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 12:15:00 | 727.20 | 720.13 | 718.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 746.25 | 749.87 | 739.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 10:00:00 | 746.25 | 749.87 | 739.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 744.70 | 748.83 | 739.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 741.00 | 748.83 | 739.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 737.75 | 746.62 | 739.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:00:00 | 737.75 | 746.62 | 739.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 737.85 | 744.86 | 739.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 737.20 | 744.86 | 739.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 739.60 | 743.94 | 739.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 739.60 | 743.94 | 739.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 746.35 | 744.43 | 740.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 738.25 | 744.43 | 740.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 728.90 | 741.32 | 739.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 729.70 | 741.32 | 739.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 728.05 | 738.67 | 738.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:30:00 | 729.85 | 738.67 | 738.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 733.85 | 737.70 | 737.99 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 742.00 | 738.07 | 737.82 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 13:15:00 | 735.55 | 737.65 | 737.68 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 742.00 | 738.10 | 737.84 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 13:15:00 | 737.00 | 737.61 | 737.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 14:15:00 | 733.65 | 736.82 | 737.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 739.50 | 730.77 | 732.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 09:15:00 | 739.50 | 730.77 | 732.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 739.50 | 730.77 | 732.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:00:00 | 739.50 | 730.77 | 732.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 738.75 | 732.37 | 733.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:30:00 | 740.40 | 732.37 | 733.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 738.45 | 734.60 | 734.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 13:15:00 | 739.45 | 735.57 | 734.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 10:15:00 | 735.70 | 737.32 | 736.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 10:15:00 | 735.70 | 737.32 | 736.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 735.70 | 737.32 | 736.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:00:00 | 735.70 | 737.32 | 736.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 737.10 | 737.28 | 736.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 12:30:00 | 740.90 | 737.63 | 736.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 13:30:00 | 739.15 | 737.87 | 736.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 742.20 | 737.65 | 736.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:45:00 | 740.45 | 737.90 | 736.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 733.85 | 737.09 | 736.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-18 10:15:00 | 733.85 | 737.09 | 736.64 | SL hit (close<static) qty=1.00 sl=735.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 730.85 | 735.84 | 736.12 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 13:15:00 | 742.60 | 737.55 | 736.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 14:15:00 | 752.45 | 740.53 | 738.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 09:15:00 | 784.50 | 784.72 | 778.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 10:15:00 | 776.75 | 783.13 | 777.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 776.75 | 783.13 | 777.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:00:00 | 776.75 | 783.13 | 777.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 782.45 | 782.99 | 778.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:30:00 | 779.95 | 782.99 | 778.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 825.40 | 829.83 | 820.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:45:00 | 821.05 | 829.83 | 820.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 824.70 | 828.01 | 822.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:45:00 | 824.15 | 828.01 | 822.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 13:15:00 | 821.60 | 826.73 | 822.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:00:00 | 821.60 | 826.73 | 822.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 823.65 | 826.11 | 822.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:45:00 | 820.55 | 826.11 | 822.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 823.20 | 825.53 | 822.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 09:30:00 | 829.15 | 825.28 | 822.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 12:15:00 | 819.70 | 823.39 | 822.24 | SL hit (close<static) qty=1.00 sl=821.15 alert=retest2 |

### Cycle 26 — SELL (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 15:15:00 | 818.30 | 821.05 | 821.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 813.60 | 819.56 | 820.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 817.00 | 815.28 | 817.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-02 15:00:00 | 817.00 | 815.28 | 817.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 816.80 | 815.58 | 817.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:15:00 | 848.45 | 815.58 | 817.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 851.30 | 822.72 | 820.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 856.40 | 844.46 | 838.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 857.85 | 860.50 | 852.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 857.85 | 860.50 | 852.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 859.30 | 860.26 | 852.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 855.60 | 860.26 | 852.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 868.70 | 872.70 | 868.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 13:45:00 | 869.95 | 872.70 | 868.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 872.70 | 872.70 | 868.81 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 12:15:00 | 863.50 | 866.92 | 867.21 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 10:15:00 | 880.20 | 869.27 | 868.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 11:15:00 | 883.45 | 872.10 | 869.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 12:15:00 | 870.65 | 871.81 | 869.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 12:15:00 | 870.65 | 871.81 | 869.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 870.65 | 871.81 | 869.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:00:00 | 870.65 | 871.81 | 869.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 868.30 | 871.11 | 869.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:00:00 | 868.30 | 871.11 | 869.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 871.80 | 871.25 | 869.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:30:00 | 870.55 | 871.25 | 869.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 870.85 | 871.17 | 869.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:15:00 | 859.10 | 871.17 | 869.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 867.05 | 870.34 | 869.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 857.55 | 870.34 | 869.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 872.85 | 870.85 | 869.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 11:15:00 | 875.70 | 870.85 | 869.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 12:00:00 | 877.50 | 872.18 | 870.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:30:00 | 878.15 | 873.71 | 872.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 11:45:00 | 876.65 | 872.42 | 871.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 870.15 | 871.96 | 871.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:00:00 | 870.15 | 871.96 | 871.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 872.45 | 872.06 | 871.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:30:00 | 871.00 | 872.06 | 871.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-17 14:15:00 | 865.60 | 870.77 | 871.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 865.60 | 870.77 | 871.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 10:15:00 | 860.35 | 867.75 | 869.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 863.30 | 860.81 | 864.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 10:15:00 | 863.30 | 860.81 | 864.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 863.30 | 860.81 | 864.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 863.30 | 860.81 | 864.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 863.20 | 861.29 | 864.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:30:00 | 864.45 | 861.29 | 864.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 857.85 | 860.60 | 863.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:30:00 | 865.15 | 860.60 | 863.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 848.80 | 843.29 | 849.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:45:00 | 850.40 | 843.29 | 849.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 845.85 | 843.80 | 849.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:30:00 | 851.25 | 843.80 | 849.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 840.10 | 840.64 | 845.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 13:15:00 | 837.30 | 841.26 | 844.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 11:15:00 | 837.60 | 832.27 | 832.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 837.60 | 832.27 | 832.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 10:15:00 | 839.10 | 835.74 | 834.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 11:15:00 | 835.60 | 835.71 | 834.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 11:15:00 | 835.60 | 835.71 | 834.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 835.60 | 835.71 | 834.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:30:00 | 835.15 | 835.71 | 834.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 829.55 | 834.48 | 833.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 13:00:00 | 829.55 | 834.48 | 833.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 826.70 | 832.93 | 833.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 15:15:00 | 824.30 | 829.55 | 831.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 836.75 | 825.62 | 827.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 836.75 | 825.62 | 827.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 836.75 | 825.62 | 827.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 10:00:00 | 836.75 | 825.62 | 827.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 835.65 | 827.63 | 828.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 11:15:00 | 832.40 | 827.63 | 828.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 12:15:00 | 834.00 | 829.38 | 828.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 12:15:00 | 834.00 | 829.38 | 828.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 849.00 | 833.30 | 830.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 14:15:00 | 861.70 | 861.74 | 854.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 15:00:00 | 861.70 | 861.74 | 854.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 853.95 | 859.90 | 854.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 10:00:00 | 853.95 | 859.90 | 854.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 853.20 | 858.56 | 854.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 10:30:00 | 851.40 | 858.56 | 854.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 858.00 | 857.54 | 854.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 13:30:00 | 858.55 | 857.71 | 855.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 864.60 | 857.87 | 855.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-20 13:15:00 | 890.15 | 891.66 | 891.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 13:15:00 | 890.15 | 891.66 | 891.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 881.25 | 888.71 | 890.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 872.75 | 867.68 | 873.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 872.75 | 867.68 | 873.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 872.75 | 867.68 | 873.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 872.75 | 867.68 | 873.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 870.35 | 868.21 | 873.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 873.70 | 868.21 | 873.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 873.55 | 869.28 | 873.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 873.55 | 869.28 | 873.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 871.05 | 869.63 | 872.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 870.00 | 869.63 | 872.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 873.05 | 870.32 | 873.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 876.60 | 870.32 | 873.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 871.80 | 870.61 | 872.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 873.65 | 870.61 | 872.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 869.55 | 870.40 | 872.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:15:00 | 869.35 | 870.40 | 872.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 11:15:00 | 875.85 | 867.53 | 869.21 | SL hit (close>static) qty=1.00 sl=874.35 alert=retest2 |

### Cycle 35 — BUY (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 13:15:00 | 877.60 | 871.06 | 870.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 880.95 | 873.04 | 871.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 11:15:00 | 872.70 | 875.15 | 873.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 11:15:00 | 872.70 | 875.15 | 873.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 872.70 | 875.15 | 873.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:45:00 | 872.50 | 875.15 | 873.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 870.50 | 874.22 | 873.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:00:00 | 870.50 | 874.22 | 873.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 874.20 | 873.85 | 873.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 870.75 | 873.85 | 873.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 875.70 | 874.22 | 873.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 920.00 | 874.22 | 873.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 11:15:00 | 880.50 | 895.47 | 895.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 880.50 | 895.47 | 895.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 10:15:00 | 875.10 | 883.18 | 888.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 889.15 | 881.22 | 884.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 889.15 | 881.22 | 884.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 889.15 | 881.22 | 884.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:45:00 | 889.80 | 881.22 | 884.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 885.30 | 882.03 | 884.79 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 13:15:00 | 893.60 | 886.61 | 886.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 14:15:00 | 901.30 | 889.55 | 887.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-10 10:15:00 | 901.40 | 903.63 | 898.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-10 11:00:00 | 901.40 | 903.63 | 898.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 900.05 | 902.45 | 898.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:45:00 | 900.50 | 902.45 | 898.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 900.05 | 901.43 | 899.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:45:00 | 898.00 | 901.43 | 899.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 906.85 | 902.51 | 900.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 914.00 | 901.73 | 900.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 14:15:00 | 908.85 | 913.20 | 912.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 15:00:00 | 908.90 | 912.34 | 912.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 15:15:00 | 908.00 | 911.47 | 911.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 15:15:00 | 908.00 | 911.47 | 911.87 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 10:15:00 | 917.90 | 913.08 | 912.55 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 908.25 | 912.43 | 912.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 15:15:00 | 905.40 | 910.60 | 911.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-23 13:15:00 | 879.20 | 871.93 | 878.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 13:15:00 | 879.20 | 871.93 | 878.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 879.20 | 871.93 | 878.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 14:00:00 | 879.20 | 871.93 | 878.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 14:15:00 | 875.20 | 872.58 | 878.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:15:00 | 871.70 | 872.58 | 878.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 09:45:00 | 872.55 | 872.79 | 877.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 10:15:00 | 873.40 | 872.79 | 877.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 11:00:00 | 874.00 | 873.04 | 877.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 861.35 | 856.21 | 861.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 861.35 | 856.21 | 861.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 860.35 | 857.04 | 861.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 863.35 | 857.04 | 861.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 859.45 | 856.76 | 859.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 10:15:00 | 867.85 | 856.76 | 859.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 863.25 | 858.06 | 859.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 10:45:00 | 868.65 | 858.06 | 859.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 853.80 | 857.21 | 859.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 863.15 | 859.86 | 859.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 863.15 | 859.86 | 859.77 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 855.95 | 859.08 | 859.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 11:15:00 | 851.90 | 857.64 | 858.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 11:15:00 | 854.20 | 849.75 | 853.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 11:15:00 | 854.20 | 849.75 | 853.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 854.20 | 849.75 | 853.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:00:00 | 854.20 | 849.75 | 853.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 854.95 | 850.79 | 853.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:45:00 | 857.65 | 850.79 | 853.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 860.30 | 852.69 | 854.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:45:00 | 861.30 | 852.69 | 854.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 859.65 | 854.09 | 854.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 14:45:00 | 861.50 | 854.09 | 854.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 870.55 | 858.04 | 856.43 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 850.85 | 856.78 | 856.81 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 860.00 | 856.35 | 856.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 865.30 | 858.96 | 857.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 14:15:00 | 978.95 | 980.61 | 967.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-13 15:00:00 | 978.95 | 980.61 | 967.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 967.60 | 976.95 | 967.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:45:00 | 963.60 | 976.95 | 967.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 968.60 | 975.28 | 967.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 10:45:00 | 968.00 | 975.28 | 967.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 975.05 | 975.24 | 968.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 981.30 | 974.21 | 971.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 10:45:00 | 983.05 | 976.93 | 973.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 10:15:00 | 1016.85 | 1027.20 | 1028.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 10:15:00 | 1016.85 | 1027.20 | 1028.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 13:15:00 | 1012.40 | 1020.49 | 1024.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 10:15:00 | 1025.50 | 1019.78 | 1022.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 10:15:00 | 1025.50 | 1019.78 | 1022.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 1025.50 | 1019.78 | 1022.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:45:00 | 1029.00 | 1019.78 | 1022.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 1024.60 | 1020.75 | 1023.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:45:00 | 1023.05 | 1020.75 | 1023.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 1022.15 | 1021.03 | 1022.92 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 1044.45 | 1025.52 | 1024.35 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 10:15:00 | 1025.60 | 1035.01 | 1035.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 11:15:00 | 1017.05 | 1031.42 | 1033.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 14:15:00 | 1027.85 | 1027.34 | 1030.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-05 15:00:00 | 1027.85 | 1027.34 | 1030.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 1025.05 | 1026.50 | 1029.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 10:30:00 | 1019.60 | 1025.71 | 1029.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 14:00:00 | 1021.50 | 1022.99 | 1024.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 10:30:00 | 1022.80 | 1022.51 | 1023.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 09:15:00 | 1041.00 | 1017.80 | 1015.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 09:15:00 | 1041.00 | 1017.80 | 1015.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 1045.50 | 1036.11 | 1030.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 1031.65 | 1037.11 | 1032.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 10:15:00 | 1031.65 | 1037.11 | 1032.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 1031.65 | 1037.11 | 1032.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:00:00 | 1031.65 | 1037.11 | 1032.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 1027.70 | 1035.23 | 1031.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 12:00:00 | 1027.70 | 1035.23 | 1031.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 1031.15 | 1034.41 | 1031.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 09:15:00 | 1037.40 | 1031.60 | 1030.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:00:00 | 1040.00 | 1033.28 | 1031.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 12:15:00 | 1026.30 | 1032.75 | 1031.99 | SL hit (close<static) qty=1.00 sl=1027.65 alert=retest2 |

### Cycle 50 — SELL (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 13:15:00 | 1023.85 | 1030.97 | 1031.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 09:15:00 | 1013.05 | 1025.24 | 1028.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 10:15:00 | 1032.30 | 1026.65 | 1028.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 10:15:00 | 1032.30 | 1026.65 | 1028.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 1032.30 | 1026.65 | 1028.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:00:00 | 1032.30 | 1026.65 | 1028.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 1029.10 | 1027.14 | 1028.75 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 15:15:00 | 1034.80 | 1030.30 | 1029.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 1038.85 | 1032.01 | 1030.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 1031.05 | 1042.67 | 1037.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 14:15:00 | 1031.05 | 1042.67 | 1037.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 1031.05 | 1042.67 | 1037.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 1031.05 | 1042.67 | 1037.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 1034.95 | 1041.13 | 1037.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 1009.10 | 1041.13 | 1037.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 1026.40 | 1034.70 | 1034.88 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 11:15:00 | 1060.95 | 1039.95 | 1037.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 12:15:00 | 1063.60 | 1044.68 | 1039.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-22 13:15:00 | 1070.90 | 1072.75 | 1060.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-22 13:30:00 | 1072.50 | 1072.75 | 1060.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 1077.80 | 1072.02 | 1063.27 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 10:15:00 | 1057.30 | 1066.29 | 1066.54 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 12:15:00 | 1070.00 | 1066.86 | 1066.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 14:15:00 | 1088.85 | 1071.71 | 1069.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 13:15:00 | 1075.30 | 1076.80 | 1073.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 13:45:00 | 1074.35 | 1076.80 | 1073.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 1078.65 | 1082.61 | 1079.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 1078.65 | 1082.61 | 1079.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 1080.70 | 1082.23 | 1079.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 1104.35 | 1082.23 | 1079.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 10:00:00 | 1087.25 | 1091.58 | 1087.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 11:15:00 | 1096.80 | 1111.92 | 1113.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 1096.80 | 1111.92 | 1113.19 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 15:15:00 | 1125.20 | 1115.51 | 1114.37 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 13:15:00 | 1104.40 | 1113.04 | 1113.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 09:15:00 | 1097.45 | 1108.21 | 1111.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 1138.25 | 1106.44 | 1107.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 1138.25 | 1106.44 | 1107.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1138.25 | 1106.44 | 1107.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 1138.25 | 1106.44 | 1107.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 10:15:00 | 1145.75 | 1114.31 | 1110.71 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 1104.20 | 1122.35 | 1123.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 1088.85 | 1105.69 | 1114.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 1106.00 | 1100.47 | 1109.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 10:45:00 | 1107.95 | 1100.47 | 1109.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 1123.85 | 1105.15 | 1110.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 1123.85 | 1105.15 | 1110.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 1113.55 | 1106.83 | 1110.81 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 15:15:00 | 1130.00 | 1116.15 | 1114.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 1143.60 | 1125.00 | 1119.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 1141.95 | 1144.22 | 1135.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 15:00:00 | 1141.95 | 1144.22 | 1135.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1143.40 | 1144.98 | 1137.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 1143.40 | 1144.98 | 1137.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1154.80 | 1146.94 | 1138.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 15:15:00 | 1165.00 | 1144.24 | 1141.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 10:00:00 | 1165.20 | 1151.76 | 1145.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 11:15:00 | 1164.15 | 1152.77 | 1146.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 11:45:00 | 1162.50 | 1153.92 | 1150.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1159.85 | 1158.11 | 1154.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 13:45:00 | 1171.25 | 1161.96 | 1157.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 09:15:00 | 1146.00 | 1156.95 | 1155.85 | SL hit (close<static) qty=1.00 sl=1152.65 alert=retest2 |

### Cycle 62 — SELL (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 12:15:00 | 1150.45 | 1155.31 | 1155.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 09:15:00 | 1075.35 | 1137.77 | 1147.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 15:15:00 | 1069.35 | 1069.26 | 1089.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 09:15:00 | 1044.80 | 1069.26 | 1089.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:15:00 | 1063.45 | 1068.78 | 1087.16 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 14:15:00 | 1010.28 | 1043.89 | 1067.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 1042.15 | 1039.99 | 1061.15 | SL hit (close>ema200) qty=0.50 sl=1039.99 alert=retest1 |

### Cycle 63 — BUY (started 2024-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 14:15:00 | 1017.75 | 1010.42 | 1009.57 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 1002.60 | 1013.31 | 1013.97 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 1019.00 | 1009.78 | 1009.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 1043.00 | 1017.43 | 1012.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 1040.60 | 1043.55 | 1034.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 14:30:00 | 1041.05 | 1043.55 | 1034.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1035.80 | 1041.50 | 1034.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:45:00 | 1030.40 | 1041.50 | 1034.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1040.35 | 1041.27 | 1035.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 14:30:00 | 1050.10 | 1044.18 | 1038.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 11:00:00 | 1047.85 | 1045.35 | 1043.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 11:30:00 | 1046.00 | 1046.55 | 1044.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 13:00:00 | 1045.95 | 1046.43 | 1044.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1044.15 | 1045.97 | 1044.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:30:00 | 1046.10 | 1045.97 | 1044.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1046.30 | 1046.04 | 1044.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:30:00 | 1043.55 | 1046.04 | 1044.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1047.95 | 1046.42 | 1044.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 1040.85 | 1046.42 | 1044.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 1047.95 | 1046.73 | 1045.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-26 09:15:00 | 1025.30 | 1042.48 | 1044.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 1025.30 | 1042.48 | 1044.04 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 14:15:00 | 1045.85 | 1041.46 | 1041.30 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1036.45 | 1040.75 | 1041.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1026.05 | 1037.81 | 1039.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 1028.05 | 1025.98 | 1031.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 1028.05 | 1025.98 | 1031.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 1026.25 | 1026.04 | 1030.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 1016.00 | 1026.04 | 1030.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 11:15:00 | 1023.25 | 1024.37 | 1028.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-02 09:15:00 | 1067.00 | 1033.65 | 1031.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 1067.00 | 1033.65 | 1031.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 1077.05 | 1042.33 | 1035.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 1078.45 | 1084.02 | 1069.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:30:00 | 1079.95 | 1084.02 | 1069.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1076.00 | 1082.41 | 1070.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 1076.00 | 1082.41 | 1070.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1058.25 | 1078.37 | 1073.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 1058.25 | 1078.37 | 1073.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 1051.80 | 1073.06 | 1071.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 1051.80 | 1073.06 | 1071.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 1042.00 | 1066.85 | 1068.65 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 12:15:00 | 1072.30 | 1065.03 | 1064.70 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 1056.30 | 1063.65 | 1064.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 1047.15 | 1060.35 | 1062.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 12:15:00 | 1005.30 | 999.91 | 1007.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 12:15:00 | 1005.30 | 999.91 | 1007.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 1005.30 | 999.91 | 1007.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:30:00 | 1005.20 | 999.91 | 1007.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 1009.60 | 1001.85 | 1007.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:30:00 | 1009.45 | 1001.85 | 1007.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 1018.20 | 1005.12 | 1008.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 15:00:00 | 1018.20 | 1005.12 | 1008.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 1022.00 | 1008.50 | 1010.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:15:00 | 1010.70 | 1008.50 | 1010.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 13:15:00 | 1012.95 | 1005.30 | 1004.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 1012.95 | 1005.30 | 1004.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 14:15:00 | 1017.95 | 1007.83 | 1005.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 10:15:00 | 1022.50 | 1022.85 | 1017.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 11:00:00 | 1022.50 | 1022.85 | 1017.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 12:15:00 | 1016.35 | 1020.74 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 13:00:00 | 1016.35 | 1020.74 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 13:15:00 | 1018.50 | 1020.29 | 1017.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 14:00:00 | 1018.50 | 1020.29 | 1017.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 1020.15 | 1020.26 | 1017.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 15:00:00 | 1020.15 | 1020.26 | 1017.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 1017.00 | 1019.61 | 1017.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 1004.05 | 1019.61 | 1017.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1007.55 | 1017.20 | 1016.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 1005.40 | 1017.20 | 1016.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 1025.50 | 1018.86 | 1017.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 11:15:00 | 1026.95 | 1018.86 | 1017.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-01 09:15:00 | 1129.65 | 1085.40 | 1067.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 11:15:00 | 1118.20 | 1121.09 | 1121.21 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 11:15:00 | 1124.00 | 1121.33 | 1121.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 14:15:00 | 1131.55 | 1123.38 | 1122.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 09:15:00 | 1122.80 | 1124.27 | 1122.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 1122.80 | 1124.27 | 1122.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 1122.80 | 1124.27 | 1122.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:45:00 | 1122.70 | 1124.27 | 1122.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 1121.85 | 1123.78 | 1122.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 1119.25 | 1123.78 | 1122.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 1118.45 | 1122.72 | 1122.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:30:00 | 1117.15 | 1122.72 | 1122.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 1120.00 | 1122.17 | 1122.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:45:00 | 1119.50 | 1122.17 | 1122.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 14:15:00 | 1106.40 | 1119.00 | 1120.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 1099.70 | 1111.73 | 1116.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1088.25 | 1087.57 | 1096.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 09:30:00 | 1094.70 | 1087.57 | 1096.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1101.00 | 1090.26 | 1096.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 1100.55 | 1090.26 | 1096.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1098.70 | 1091.95 | 1096.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 1100.35 | 1091.95 | 1096.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 1109.50 | 1097.60 | 1098.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 1109.50 | 1097.60 | 1098.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 14:15:00 | 1110.50 | 1100.18 | 1099.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 10:15:00 | 1111.90 | 1103.98 | 1101.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 1095.00 | 1108.37 | 1105.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 1095.00 | 1108.37 | 1105.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1095.00 | 1108.37 | 1105.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:30:00 | 1093.65 | 1108.37 | 1105.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 1084.35 | 1103.56 | 1103.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 13:15:00 | 1078.35 | 1086.66 | 1092.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 09:15:00 | 1087.10 | 1085.06 | 1090.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 1087.10 | 1085.06 | 1090.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1087.10 | 1085.06 | 1090.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:00:00 | 1087.10 | 1085.06 | 1090.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 1088.00 | 1085.42 | 1089.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:45:00 | 1087.85 | 1085.42 | 1089.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 1088.95 | 1085.66 | 1088.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:00:00 | 1088.95 | 1085.66 | 1088.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 1094.00 | 1087.33 | 1089.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:30:00 | 1094.55 | 1087.33 | 1089.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 1090.00 | 1087.86 | 1089.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:15:00 | 1092.75 | 1087.86 | 1089.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 1093.00 | 1089.25 | 1089.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:45:00 | 1093.95 | 1089.25 | 1089.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 1087.05 | 1088.81 | 1089.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 12:15:00 | 1084.95 | 1088.81 | 1089.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 14:45:00 | 1084.95 | 1087.87 | 1088.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 09:30:00 | 1082.50 | 1086.41 | 1088.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 10:45:00 | 1081.75 | 1086.03 | 1087.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1089.65 | 1086.75 | 1087.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:45:00 | 1090.15 | 1086.75 | 1087.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 1092.70 | 1087.94 | 1088.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 13:00:00 | 1092.70 | 1087.94 | 1088.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-25 13:15:00 | 1101.75 | 1090.70 | 1089.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 13:15:00 | 1101.75 | 1090.70 | 1089.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 1104.90 | 1093.54 | 1090.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 1154.10 | 1158.48 | 1144.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 15:00:00 | 1154.10 | 1158.48 | 1144.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1154.05 | 1156.24 | 1145.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 14:30:00 | 1161.95 | 1155.46 | 1149.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 1167.70 | 1155.93 | 1149.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 1161.25 | 1156.63 | 1150.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:45:00 | 1162.15 | 1156.88 | 1151.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 1149.35 | 1155.37 | 1151.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:30:00 | 1152.15 | 1155.37 | 1151.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 1152.45 | 1154.79 | 1151.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 10:45:00 | 1160.00 | 1153.81 | 1152.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:45:00 | 1167.60 | 1155.65 | 1153.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 1137.20 | 1155.63 | 1154.57 | SL hit (close<static) qty=1.00 sl=1145.60 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 1114.55 | 1147.41 | 1150.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 1103.85 | 1138.70 | 1146.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 11:15:00 | 1130.75 | 1126.74 | 1134.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 11:45:00 | 1131.65 | 1126.74 | 1134.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1142.60 | 1129.91 | 1135.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 1142.10 | 1129.91 | 1135.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 1132.10 | 1130.35 | 1135.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 1127.45 | 1131.34 | 1134.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 09:15:00 | 1146.50 | 1134.37 | 1135.78 | SL hit (close>static) qty=1.00 sl=1145.50 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1141.50 | 1133.70 | 1132.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 1167.50 | 1142.41 | 1137.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 11:15:00 | 1158.60 | 1159.03 | 1149.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 11:30:00 | 1159.85 | 1159.03 | 1149.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 1160.50 | 1160.36 | 1154.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 1158.80 | 1160.36 | 1154.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1157.05 | 1160.51 | 1157.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 1152.25 | 1160.51 | 1157.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 1151.20 | 1158.65 | 1156.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 1155.05 | 1158.65 | 1156.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 1150.65 | 1157.05 | 1155.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:15:00 | 1151.90 | 1157.05 | 1155.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1156.15 | 1156.87 | 1155.97 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 1149.15 | 1154.51 | 1155.01 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 1160.10 | 1156.17 | 1155.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 1173.35 | 1159.61 | 1157.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 13:15:00 | 1167.50 | 1169.50 | 1163.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 14:00:00 | 1167.50 | 1169.50 | 1163.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 1163.95 | 1168.39 | 1163.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 1163.95 | 1168.39 | 1163.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 1171.00 | 1168.91 | 1164.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 1193.10 | 1168.91 | 1164.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 1204.25 | 1219.83 | 1221.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 1204.25 | 1219.83 | 1221.59 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 1224.70 | 1217.35 | 1217.11 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1211.60 | 1216.65 | 1217.02 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 11:15:00 | 1219.05 | 1217.48 | 1217.35 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 13:15:00 | 1213.20 | 1216.95 | 1217.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 1200.15 | 1213.59 | 1215.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1228.80 | 1197.82 | 1201.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 1228.80 | 1197.82 | 1201.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1228.80 | 1197.82 | 1201.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 1228.80 | 1197.82 | 1201.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 1221.45 | 1206.69 | 1205.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 1230.00 | 1217.62 | 1211.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1187.75 | 1212.95 | 1210.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1187.75 | 1212.95 | 1210.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1187.75 | 1212.95 | 1210.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1187.75 | 1212.95 | 1210.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1170.40 | 1204.44 | 1206.93 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 1236.55 | 1208.13 | 1207.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 1255.40 | 1217.58 | 1211.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 1252.45 | 1252.61 | 1237.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:45:00 | 1259.20 | 1252.61 | 1237.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1243.85 | 1260.90 | 1254.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 1243.85 | 1260.90 | 1254.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 1244.20 | 1257.56 | 1253.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:30:00 | 1240.00 | 1257.56 | 1253.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 1241.90 | 1251.71 | 1251.48 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 13:15:00 | 1245.75 | 1250.52 | 1250.96 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 10:15:00 | 1259.90 | 1251.83 | 1251.30 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1246.65 | 1252.59 | 1252.68 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 13:15:00 | 1254.85 | 1251.68 | 1251.66 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 1235.35 | 1251.36 | 1252.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 10:15:00 | 1231.70 | 1247.43 | 1250.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 12:15:00 | 1247.25 | 1246.54 | 1249.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 12:15:00 | 1247.25 | 1246.54 | 1249.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 1247.25 | 1246.54 | 1249.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:00:00 | 1247.25 | 1246.54 | 1249.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 1242.60 | 1245.75 | 1248.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:45:00 | 1250.00 | 1245.75 | 1248.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1226.00 | 1240.49 | 1245.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 14:45:00 | 1225.15 | 1231.84 | 1238.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1255.10 | 1240.90 | 1239.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1255.10 | 1240.90 | 1239.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 10:15:00 | 1258.20 | 1244.36 | 1241.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 1240.55 | 1244.74 | 1242.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 14:15:00 | 1240.55 | 1244.74 | 1242.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1240.55 | 1244.74 | 1242.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 1240.55 | 1244.74 | 1242.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1240.00 | 1243.79 | 1242.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1226.75 | 1243.79 | 1242.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 1227.75 | 1240.58 | 1240.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 1208.10 | 1225.41 | 1232.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 11:15:00 | 1226.90 | 1224.10 | 1230.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:00:00 | 1226.90 | 1224.10 | 1230.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1197.40 | 1203.38 | 1211.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 1197.40 | 1203.38 | 1211.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1206.80 | 1197.66 | 1204.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 1206.80 | 1197.66 | 1204.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1203.50 | 1198.83 | 1204.22 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1218.65 | 1206.75 | 1206.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1233.85 | 1216.59 | 1212.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 1310.50 | 1320.26 | 1306.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 10:15:00 | 1310.50 | 1320.26 | 1306.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1310.50 | 1320.26 | 1306.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1309.10 | 1320.26 | 1306.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 1317.40 | 1320.82 | 1315.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:30:00 | 1316.10 | 1320.82 | 1315.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1318.00 | 1320.25 | 1316.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 1318.00 | 1320.25 | 1316.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1318.70 | 1319.94 | 1316.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 1323.95 | 1319.94 | 1316.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1317.10 | 1319.38 | 1316.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 1326.15 | 1321.70 | 1318.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 1327.00 | 1321.70 | 1318.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 12:15:00 | 1327.30 | 1352.25 | 1354.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 1327.30 | 1352.25 | 1354.47 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 1354.45 | 1347.49 | 1346.62 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1342.10 | 1346.68 | 1346.73 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 1356.20 | 1348.58 | 1347.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 14:15:00 | 1358.05 | 1350.47 | 1348.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 10:15:00 | 1352.60 | 1352.64 | 1350.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 11:00:00 | 1352.60 | 1352.64 | 1350.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 1353.15 | 1352.74 | 1350.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:00:00 | 1357.05 | 1352.97 | 1351.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:45:00 | 1359.80 | 1356.91 | 1353.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 13:15:00 | 1413.60 | 1425.82 | 1425.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 1413.60 | 1425.82 | 1425.87 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 1429.10 | 1425.94 | 1425.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 1430.05 | 1426.76 | 1426.08 | Break + close above crossover candle high |

### Cycle 106 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 1414.70 | 1424.35 | 1425.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1400.20 | 1419.52 | 1422.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1431.95 | 1419.15 | 1421.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 1431.95 | 1419.15 | 1421.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1431.95 | 1419.15 | 1421.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 1434.65 | 1419.15 | 1421.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 1443.35 | 1423.99 | 1423.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 13:15:00 | 1456.45 | 1433.91 | 1428.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 1467.30 | 1468.51 | 1454.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 09:30:00 | 1470.60 | 1468.51 | 1454.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 1454.35 | 1465.68 | 1454.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:30:00 | 1456.95 | 1465.68 | 1454.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 1460.25 | 1464.60 | 1455.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 12:45:00 | 1465.00 | 1463.78 | 1455.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 13:45:00 | 1464.95 | 1464.33 | 1456.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 15:15:00 | 1442.00 | 1457.96 | 1455.00 | SL hit (close<static) qty=1.00 sl=1452.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 1479.35 | 1482.64 | 1482.83 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 1490.20 | 1484.15 | 1483.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 1499.90 | 1488.42 | 1485.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 1505.75 | 1523.58 | 1515.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1505.75 | 1523.58 | 1515.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1505.75 | 1523.58 | 1515.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 1505.75 | 1523.58 | 1515.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1507.00 | 1520.27 | 1514.29 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 13:15:00 | 1502.65 | 1510.24 | 1510.65 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 1514.00 | 1511.19 | 1511.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 1519.30 | 1512.98 | 1511.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 15:15:00 | 1534.35 | 1535.48 | 1528.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 09:15:00 | 1538.75 | 1535.48 | 1528.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 1557.45 | 1562.46 | 1557.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 1557.45 | 1562.46 | 1557.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1563.80 | 1562.73 | 1558.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:45:00 | 1568.25 | 1563.95 | 1559.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 12:00:00 | 1568.20 | 1565.82 | 1561.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 15:15:00 | 1568.05 | 1566.04 | 1562.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 10:30:00 | 1571.95 | 1566.15 | 1563.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1559.25 | 1564.77 | 1563.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 1559.25 | 1564.77 | 1563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1556.65 | 1563.15 | 1562.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 1557.90 | 1563.15 | 1562.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-02 13:15:00 | 1554.00 | 1561.32 | 1561.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 1554.00 | 1561.32 | 1561.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 15:15:00 | 1550.20 | 1557.86 | 1560.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1562.30 | 1558.75 | 1560.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 1562.30 | 1558.75 | 1560.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1562.30 | 1558.75 | 1560.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 1562.30 | 1558.75 | 1560.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 1555.00 | 1558.00 | 1559.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:45:00 | 1553.95 | 1558.35 | 1559.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:45:00 | 1548.75 | 1551.05 | 1553.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:00:00 | 1553.25 | 1551.49 | 1553.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 1565.15 | 1527.59 | 1526.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 1565.15 | 1527.59 | 1526.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1582.40 | 1558.37 | 1545.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 11:15:00 | 1558.40 | 1559.70 | 1548.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 12:00:00 | 1558.40 | 1559.70 | 1548.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1560.45 | 1564.81 | 1558.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 1560.45 | 1564.81 | 1558.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1562.30 | 1564.31 | 1558.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:30:00 | 1554.80 | 1564.31 | 1558.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1567.15 | 1564.87 | 1559.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 1561.40 | 1564.87 | 1559.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1554.50 | 1562.80 | 1559.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1554.50 | 1562.80 | 1559.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1559.45 | 1562.13 | 1559.13 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 1550.00 | 1557.60 | 1557.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 1531.15 | 1552.31 | 1555.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1540.00 | 1539.96 | 1546.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1540.00 | 1539.96 | 1546.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1540.00 | 1539.96 | 1546.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1536.60 | 1539.96 | 1546.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 1511.90 | 1501.62 | 1500.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 1511.90 | 1501.62 | 1500.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 12:15:00 | 1518.55 | 1505.01 | 1502.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 11:15:00 | 1508.50 | 1514.05 | 1509.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 11:15:00 | 1508.50 | 1514.05 | 1509.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1508.50 | 1514.05 | 1509.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 1508.50 | 1514.05 | 1509.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 1501.35 | 1511.51 | 1508.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 1501.35 | 1511.51 | 1508.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 1503.55 | 1509.92 | 1508.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:30:00 | 1496.30 | 1509.92 | 1508.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1508.00 | 1510.64 | 1508.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1503.40 | 1510.64 | 1508.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1497.85 | 1508.08 | 1507.83 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 1496.55 | 1505.77 | 1506.81 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 1510.50 | 1507.53 | 1507.25 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 1494.95 | 1505.04 | 1506.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1472.65 | 1497.13 | 1502.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 1451.10 | 1446.40 | 1463.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 09:30:00 | 1453.90 | 1446.40 | 1463.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1460.65 | 1450.78 | 1459.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 1460.65 | 1450.78 | 1459.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1459.95 | 1452.61 | 1459.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1447.20 | 1452.61 | 1459.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1466.30 | 1455.35 | 1460.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 1473.45 | 1455.35 | 1460.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1466.10 | 1457.50 | 1460.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:30:00 | 1473.00 | 1457.50 | 1460.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 12:15:00 | 1477.10 | 1465.63 | 1464.22 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 1463.65 | 1464.43 | 1464.52 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 13:15:00 | 1467.15 | 1464.97 | 1464.76 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 1459.55 | 1463.92 | 1464.32 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 09:15:00 | 1472.40 | 1465.62 | 1465.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 10:15:00 | 1482.65 | 1469.03 | 1466.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 1494.00 | 1500.58 | 1490.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 1494.00 | 1500.58 | 1490.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1494.00 | 1500.58 | 1490.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 1496.00 | 1500.58 | 1490.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 1483.40 | 1497.14 | 1489.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 1483.40 | 1497.14 | 1489.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1483.55 | 1494.42 | 1489.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 1475.50 | 1494.42 | 1489.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 1460.00 | 1483.26 | 1484.81 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 12:15:00 | 1495.25 | 1485.40 | 1484.56 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 1474.85 | 1485.15 | 1485.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 11:15:00 | 1471.05 | 1482.33 | 1483.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 1491.45 | 1482.34 | 1483.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 13:15:00 | 1491.45 | 1482.34 | 1483.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1491.45 | 1482.34 | 1483.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 1491.45 | 1482.34 | 1483.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1487.20 | 1483.31 | 1483.86 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 1490.00 | 1484.65 | 1484.42 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 1475.40 | 1483.45 | 1484.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 1473.15 | 1481.39 | 1483.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 1481.00 | 1479.45 | 1481.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 15:00:00 | 1481.00 | 1479.45 | 1481.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 1480.00 | 1479.56 | 1481.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 1481.30 | 1479.56 | 1481.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1472.90 | 1478.23 | 1480.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 1467.00 | 1476.38 | 1479.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 15:15:00 | 1463.85 | 1475.02 | 1477.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 1465.15 | 1471.26 | 1474.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1491.00 | 1476.10 | 1475.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1491.00 | 1476.10 | 1475.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 10:15:00 | 1493.95 | 1479.67 | 1477.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 12:15:00 | 1476.00 | 1480.52 | 1478.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 12:15:00 | 1476.00 | 1480.52 | 1478.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 1476.00 | 1480.52 | 1478.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 1476.00 | 1480.52 | 1478.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1463.45 | 1477.10 | 1476.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 1463.45 | 1477.10 | 1476.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 1457.45 | 1473.17 | 1475.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1444.00 | 1458.54 | 1465.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 11:15:00 | 1461.65 | 1455.43 | 1461.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 11:15:00 | 1461.65 | 1455.43 | 1461.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1461.65 | 1455.43 | 1461.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1464.20 | 1455.43 | 1461.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1452.40 | 1454.82 | 1460.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 1443.15 | 1454.86 | 1460.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 1447.25 | 1440.88 | 1446.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:15:00 | 1370.99 | 1393.55 | 1399.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:15:00 | 1374.89 | 1393.55 | 1399.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-05 14:15:00 | 1396.00 | 1386.52 | 1393.04 | SL hit (close>ema200) qty=0.50 sl=1386.52 alert=retest2 |

### Cycle 131 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 1248.00 | 1236.63 | 1235.55 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 1221.45 | 1234.29 | 1234.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 11:15:00 | 1218.70 | 1225.88 | 1228.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 1226.40 | 1225.99 | 1228.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 12:30:00 | 1226.65 | 1225.99 | 1228.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1228.50 | 1225.15 | 1227.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 1231.00 | 1225.15 | 1227.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1221.55 | 1224.43 | 1226.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 1225.65 | 1224.43 | 1226.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1233.45 | 1226.23 | 1227.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:45:00 | 1235.40 | 1226.23 | 1227.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1234.65 | 1227.92 | 1227.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:45:00 | 1235.55 | 1227.92 | 1227.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 13:15:00 | 1237.05 | 1229.74 | 1228.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 15:15:00 | 1244.00 | 1234.02 | 1230.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 10:15:00 | 1243.30 | 1251.74 | 1245.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 10:15:00 | 1243.30 | 1251.74 | 1245.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1243.30 | 1251.74 | 1245.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 1243.30 | 1251.74 | 1245.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1251.35 | 1251.66 | 1245.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 1245.00 | 1251.66 | 1245.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1251.75 | 1253.87 | 1249.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 1248.85 | 1253.87 | 1249.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1252.85 | 1253.67 | 1249.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:45:00 | 1253.90 | 1253.67 | 1249.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 1248.95 | 1252.72 | 1249.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:30:00 | 1247.35 | 1252.72 | 1249.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 1249.25 | 1252.03 | 1249.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:45:00 | 1248.90 | 1252.03 | 1249.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1252.95 | 1252.21 | 1249.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 15:00:00 | 1268.75 | 1255.52 | 1251.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 1241.90 | 1254.15 | 1251.73 | SL hit (close<static) qty=1.00 sl=1248.05 alert=retest2 |

### Cycle 134 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 1239.15 | 1251.47 | 1251.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 1238.30 | 1244.96 | 1246.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 10:15:00 | 1242.70 | 1242.54 | 1244.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 10:15:00 | 1242.70 | 1242.54 | 1244.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1242.70 | 1242.54 | 1244.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 1246.90 | 1242.54 | 1244.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1243.50 | 1242.73 | 1244.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:00:00 | 1243.50 | 1242.73 | 1244.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1225.30 | 1208.26 | 1213.81 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1245.75 | 1221.13 | 1218.78 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 1211.50 | 1222.31 | 1223.57 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 1238.10 | 1225.47 | 1224.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 1252.45 | 1230.86 | 1227.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 14:15:00 | 1254.70 | 1256.51 | 1246.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 1254.70 | 1256.51 | 1246.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1240.15 | 1257.42 | 1253.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1240.15 | 1257.42 | 1253.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1241.00 | 1254.14 | 1251.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:45:00 | 1246.95 | 1251.91 | 1251.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:45:00 | 1253.50 | 1252.23 | 1251.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 12:15:00 | 1251.80 | 1254.91 | 1255.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 1251.80 | 1254.91 | 1255.20 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1264.30 | 1256.91 | 1256.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1272.70 | 1262.28 | 1258.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1337.20 | 1340.39 | 1324.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:00:00 | 1337.20 | 1340.39 | 1324.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1332.25 | 1341.09 | 1332.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 1332.25 | 1341.09 | 1332.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1325.05 | 1337.88 | 1332.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 1325.05 | 1337.88 | 1332.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1326.50 | 1335.61 | 1331.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 1323.30 | 1335.61 | 1331.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 1319.30 | 1328.21 | 1328.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1312.60 | 1323.93 | 1326.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1322.60 | 1316.25 | 1319.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 1322.60 | 1316.25 | 1319.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1322.60 | 1316.25 | 1319.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 1323.05 | 1316.25 | 1319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1323.25 | 1317.65 | 1320.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:15:00 | 1317.25 | 1317.65 | 1320.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 14:15:00 | 1251.39 | 1270.78 | 1284.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 13:15:00 | 1185.53 | 1222.87 | 1251.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 141 — BUY (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 14:15:00 | 1177.05 | 1164.13 | 1164.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 10:15:00 | 1192.50 | 1173.71 | 1168.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 12:15:00 | 1196.05 | 1197.56 | 1187.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 12:30:00 | 1195.30 | 1197.56 | 1187.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1185.40 | 1194.97 | 1188.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 1185.40 | 1194.97 | 1188.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1187.00 | 1193.38 | 1187.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1181.45 | 1193.38 | 1187.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1175.05 | 1189.71 | 1186.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1177.15 | 1189.71 | 1186.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1175.55 | 1186.88 | 1185.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:15:00 | 1173.50 | 1186.88 | 1185.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 1175.65 | 1184.63 | 1184.86 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 1192.10 | 1185.64 | 1185.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 1231.70 | 1195.73 | 1189.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 1221.00 | 1230.42 | 1214.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:00:00 | 1221.00 | 1230.42 | 1214.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 1217.10 | 1226.76 | 1217.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 1217.10 | 1226.76 | 1217.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 1214.00 | 1224.21 | 1216.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:30:00 | 1213.70 | 1224.21 | 1216.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1209.60 | 1221.29 | 1216.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 1215.10 | 1221.29 | 1216.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1182.00 | 1211.94 | 1212.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 1170.15 | 1198.92 | 1206.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 11:15:00 | 1151.80 | 1151.31 | 1166.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 12:00:00 | 1151.80 | 1151.31 | 1166.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1179.75 | 1159.32 | 1164.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 1179.10 | 1159.32 | 1164.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1176.30 | 1162.72 | 1165.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:15:00 | 1183.05 | 1162.72 | 1165.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 1187.40 | 1171.08 | 1169.34 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 1164.70 | 1170.79 | 1171.09 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 1173.95 | 1171.42 | 1171.35 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 15:15:00 | 1170.40 | 1171.22 | 1171.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 09:15:00 | 1164.70 | 1169.91 | 1170.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1163.00 | 1136.02 | 1144.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1163.00 | 1136.02 | 1144.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1163.00 | 1136.02 | 1144.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1165.80 | 1136.02 | 1144.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1167.55 | 1142.32 | 1146.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 1167.55 | 1142.32 | 1146.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 1165.15 | 1151.05 | 1150.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 1171.60 | 1157.66 | 1153.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1190.60 | 1194.00 | 1183.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:45:00 | 1190.75 | 1194.00 | 1183.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1212.00 | 1197.06 | 1187.62 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1163.85 | 1185.27 | 1187.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1153.00 | 1173.31 | 1179.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1151.95 | 1148.03 | 1158.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 1151.95 | 1148.03 | 1158.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1204.45 | 1157.58 | 1159.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1204.45 | 1157.58 | 1159.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1197.90 | 1165.64 | 1163.01 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1137.00 | 1164.50 | 1166.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 1126.25 | 1151.58 | 1159.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 1143.75 | 1141.78 | 1151.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 09:45:00 | 1147.00 | 1141.78 | 1151.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1156.95 | 1144.81 | 1152.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 1160.00 | 1144.81 | 1152.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 1158.40 | 1147.53 | 1152.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 1158.40 | 1147.53 | 1152.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 1170.20 | 1157.66 | 1156.49 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 1147.05 | 1154.73 | 1155.50 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 1163.50 | 1156.98 | 1156.25 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 09:15:00 | 1129.30 | 1152.57 | 1154.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 1050.00 | 1060.59 | 1072.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1065.00 | 1056.60 | 1067.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 1065.00 | 1056.60 | 1067.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 1064.00 | 1058.08 | 1067.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 1063.00 | 1058.08 | 1067.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1062.05 | 1058.87 | 1066.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:15:00 | 1060.30 | 1058.87 | 1066.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1077.60 | 1058.49 | 1061.12 | SL hit (close>static) qty=1.00 sl=1067.65 alert=retest2 |

### Cycle 157 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1082.55 | 1063.30 | 1063.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1090.65 | 1068.77 | 1065.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1105.70 | 1106.79 | 1094.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 13:15:00 | 1093.60 | 1103.42 | 1096.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 1093.60 | 1103.42 | 1096.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 1093.50 | 1103.42 | 1096.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 1097.25 | 1102.19 | 1096.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:45:00 | 1104.50 | 1103.38 | 1098.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 1088.20 | 1099.54 | 1098.83 | SL hit (close<static) qty=1.00 sl=1092.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1093.50 | 1098.34 | 1098.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 1083.30 | 1091.24 | 1094.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1108.90 | 1094.46 | 1094.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 1108.90 | 1094.46 | 1094.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1108.90 | 1094.46 | 1094.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1108.90 | 1094.46 | 1094.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 1109.00 | 1097.37 | 1096.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 1111.15 | 1100.95 | 1099.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1185.00 | 1201.82 | 1190.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 1185.00 | 1201.82 | 1190.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1185.00 | 1201.82 | 1190.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 1179.45 | 1201.82 | 1190.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1186.35 | 1198.73 | 1190.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1179.80 | 1198.73 | 1190.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1187.60 | 1194.54 | 1189.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 1191.25 | 1194.54 | 1189.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 1192.20 | 1192.88 | 1190.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 1181.15 | 1190.53 | 1189.57 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 1181.35 | 1188.70 | 1188.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 1172.85 | 1185.53 | 1187.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1158.00 | 1156.86 | 1168.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:00:00 | 1158.00 | 1156.86 | 1168.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1163.80 | 1158.25 | 1167.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1163.80 | 1158.25 | 1167.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1165.00 | 1159.60 | 1167.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1167.90 | 1159.60 | 1167.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1158.05 | 1159.29 | 1166.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 1153.35 | 1160.45 | 1165.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 1155.00 | 1160.86 | 1164.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:45:00 | 1152.30 | 1159.79 | 1163.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:00:00 | 1154.85 | 1158.80 | 1162.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1149.85 | 1145.82 | 1152.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 1204.95 | 1162.03 | 1157.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 1204.95 | 1162.03 | 1157.62 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1065.20 | 1153.18 | 1158.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 1055.40 | 1085.05 | 1098.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1092.20 | 1071.61 | 1084.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1092.20 | 1071.61 | 1084.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1092.20 | 1071.61 | 1084.07 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1120.00 | 1090.63 | 1088.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 1124.80 | 1102.51 | 1094.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 1129.00 | 1130.89 | 1117.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:30:00 | 1131.70 | 1130.89 | 1117.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1230.00 | 1249.86 | 1232.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1230.00 | 1249.86 | 1232.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1220.00 | 1243.89 | 1231.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1220.00 | 1243.89 | 1231.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1220.80 | 1239.27 | 1230.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 1215.00 | 1239.27 | 1230.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1220.60 | 1229.29 | 1227.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:15:00 | 1217.10 | 1229.29 | 1227.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1240.50 | 1230.92 | 1228.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 1244.10 | 1230.92 | 1228.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 15:00:00 | 1247.20 | 1241.70 | 1235.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 1208.80 | 1236.45 | 1233.96 | SL hit (close<static) qty=1.00 sl=1226.60 alert=retest2 |

### Cycle 164 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 1220.40 | 1231.69 | 1232.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 1216.30 | 1228.61 | 1230.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 1228.30 | 1221.99 | 1226.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1228.30 | 1221.99 | 1226.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1228.30 | 1221.99 | 1226.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 1207.70 | 1222.68 | 1224.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:00:00 | 1209.30 | 1220.00 | 1223.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:45:00 | 1211.50 | 1218.46 | 1222.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 1210.10 | 1218.46 | 1222.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1233.20 | 1219.28 | 1221.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1233.20 | 1219.28 | 1221.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1232.50 | 1221.92 | 1222.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 1249.60 | 1227.46 | 1225.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 1249.60 | 1227.46 | 1225.03 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1216.30 | 1225.89 | 1226.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 1207.00 | 1216.33 | 1221.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 1164.20 | 1159.41 | 1174.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 1164.20 | 1159.41 | 1174.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1198.40 | 1168.15 | 1174.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 1198.40 | 1168.15 | 1174.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 11:15:00 | 1199.60 | 1174.44 | 1176.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 12:00:00 | 1199.60 | 1174.44 | 1176.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1205.90 | 1180.73 | 1179.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1243.00 | 1202.40 | 1190.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 1216.40 | 1220.34 | 1209.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 1216.40 | 1220.34 | 1209.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1216.60 | 1219.14 | 1210.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:30:00 | 1213.20 | 1219.14 | 1210.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 1212.50 | 1217.81 | 1210.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 14:45:00 | 1217.10 | 1218.35 | 1211.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:45:00 | 1219.00 | 1219.44 | 1213.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:15:00 | 1217.80 | 1218.15 | 1213.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:45:00 | 1218.30 | 1217.92 | 1213.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1216.90 | 1218.98 | 1215.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 1216.90 | 1218.98 | 1215.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 1220.60 | 1219.30 | 1215.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 1215.50 | 1219.30 | 1215.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1212.10 | 1217.86 | 1215.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-16 11:15:00 | 1206.70 | 1214.47 | 1214.30 | SL hit (close<static) qty=1.00 sl=1208.70 alert=retest2 |

### Cycle 168 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 1201.00 | 1211.77 | 1213.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1197.60 | 1206.78 | 1209.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1219.00 | 1205.44 | 1207.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1219.00 | 1205.44 | 1207.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1219.00 | 1205.44 | 1207.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1217.30 | 1205.44 | 1207.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1215.00 | 1207.35 | 1208.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 1219.70 | 1207.35 | 1208.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1194.40 | 1203.23 | 1205.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:30:00 | 1188.10 | 1197.33 | 1202.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 1187.10 | 1191.10 | 1197.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:45:00 | 1184.40 | 1190.21 | 1194.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 1185.90 | 1189.39 | 1194.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1196.70 | 1190.85 | 1194.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 1196.70 | 1190.85 | 1194.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1191.30 | 1190.94 | 1194.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1201.40 | 1190.94 | 1194.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1194.00 | 1191.55 | 1194.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:15:00 | 1193.50 | 1191.55 | 1194.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:30:00 | 1193.60 | 1190.73 | 1193.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:00:00 | 1193.40 | 1188.02 | 1190.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 1193.50 | 1187.39 | 1190.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1186.50 | 1185.65 | 1188.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 1190.20 | 1185.65 | 1188.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1185.40 | 1185.60 | 1188.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1167.50 | 1187.11 | 1188.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.83 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.92 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.73 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.83 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1145.40 | 1141.05 | 1148.26 | SL hit (close>ema200) qty=0.50 sl=1141.05 alert=retest2 |

### Cycle 169 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1156.70 | 1146.26 | 1145.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1159.40 | 1152.99 | 1149.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1153.60 | 1154.61 | 1151.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 1153.60 | 1154.61 | 1151.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1153.60 | 1154.61 | 1151.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1152.20 | 1154.61 | 1151.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1160.00 | 1155.70 | 1152.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 1163.50 | 1157.26 | 1153.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:15:00 | 1161.10 | 1158.01 | 1154.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 1162.70 | 1162.62 | 1159.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 1164.90 | 1163.08 | 1160.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1165.20 | 1167.58 | 1164.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 1165.20 | 1167.58 | 1164.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1159.80 | 1166.02 | 1164.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 1159.10 | 1166.02 | 1164.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1163.30 | 1165.48 | 1164.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 1163.20 | 1165.48 | 1164.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1157.20 | 1163.82 | 1163.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1157.20 | 1163.82 | 1163.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 1160.10 | 1163.08 | 1163.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 1160.10 | 1163.08 | 1163.10 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 15:15:00 | 1166.80 | 1163.82 | 1163.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 1174.10 | 1165.88 | 1164.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 1165.30 | 1166.37 | 1164.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 1165.30 | 1166.37 | 1164.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1165.30 | 1166.37 | 1164.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1165.30 | 1166.37 | 1164.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1163.50 | 1165.80 | 1164.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1162.00 | 1165.80 | 1164.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1153.60 | 1163.36 | 1163.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1145.70 | 1157.28 | 1160.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1151.00 | 1149.40 | 1154.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:30:00 | 1149.60 | 1149.40 | 1154.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1142.00 | 1147.68 | 1152.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 1134.50 | 1144.13 | 1148.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1112.30 | 1100.89 | 1100.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1112.30 | 1100.89 | 1100.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 1117.70 | 1110.40 | 1106.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1125.50 | 1127.15 | 1121.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1125.50 | 1127.15 | 1121.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1121.30 | 1125.41 | 1121.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1121.30 | 1125.41 | 1121.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1125.00 | 1125.33 | 1121.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1153.60 | 1125.33 | 1121.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 1119.00 | 1130.28 | 1127.55 | SL hit (close<static) qty=1.00 sl=1120.60 alert=retest2 |

### Cycle 174 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1119.50 | 1125.70 | 1125.79 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 1131.90 | 1126.83 | 1126.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 15:15:00 | 1136.20 | 1128.71 | 1127.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 10:15:00 | 1146.30 | 1151.39 | 1143.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:45:00 | 1145.00 | 1151.39 | 1143.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1147.50 | 1175.99 | 1172.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1147.50 | 1175.99 | 1172.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 1136.80 | 1168.15 | 1169.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 1131.60 | 1141.55 | 1149.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1133.10 | 1129.62 | 1136.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 1133.10 | 1129.62 | 1136.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1137.70 | 1129.94 | 1134.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1140.40 | 1129.94 | 1134.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1139.00 | 1131.75 | 1135.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 1139.00 | 1131.75 | 1135.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 1141.50 | 1134.70 | 1135.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 1143.20 | 1134.70 | 1135.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1138.00 | 1136.80 | 1136.74 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 1135.70 | 1136.58 | 1136.65 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1138.90 | 1137.04 | 1136.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 1152.90 | 1140.21 | 1138.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 1153.90 | 1155.59 | 1151.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 13:45:00 | 1155.00 | 1155.59 | 1151.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1148.00 | 1153.61 | 1151.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1147.40 | 1153.61 | 1151.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1144.20 | 1151.73 | 1150.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1145.40 | 1151.73 | 1150.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 1143.70 | 1149.41 | 1149.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 1141.10 | 1147.75 | 1149.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1145.70 | 1144.15 | 1146.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 1145.70 | 1144.15 | 1146.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1145.70 | 1144.15 | 1146.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1145.70 | 1144.15 | 1146.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1144.90 | 1144.30 | 1146.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1148.20 | 1144.30 | 1146.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1117.50 | 1115.43 | 1120.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 1112.00 | 1115.43 | 1120.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 1115.90 | 1118.66 | 1120.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:00:00 | 1112.50 | 1117.65 | 1119.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 1125.40 | 1120.80 | 1120.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 14:15:00 | 1125.40 | 1120.80 | 1120.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 09:15:00 | 1132.20 | 1122.98 | 1121.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 15:15:00 | 1124.90 | 1125.12 | 1123.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:15:00 | 1133.90 | 1125.12 | 1123.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1137.10 | 1127.52 | 1124.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:15:00 | 1146.50 | 1130.30 | 1126.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1103.20 | 1139.98 | 1144.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1103.20 | 1139.98 | 1144.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1085.50 | 1110.45 | 1127.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1094.90 | 1091.42 | 1108.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 1094.90 | 1091.42 | 1108.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1076.40 | 1088.60 | 1101.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 1066.40 | 1081.89 | 1096.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:15:00 | 1066.40 | 1078.34 | 1092.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:15:00 | 1066.30 | 1072.35 | 1079.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1061.10 | 1065.30 | 1071.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1052.00 | 1049.93 | 1055.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1055.50 | 1049.93 | 1055.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1056.10 | 1051.26 | 1055.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 1056.10 | 1051.26 | 1055.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1054.70 | 1051.94 | 1055.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 1056.20 | 1051.94 | 1055.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1059.60 | 1053.48 | 1055.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:45:00 | 1059.80 | 1053.48 | 1055.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1063.60 | 1055.50 | 1056.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 1063.60 | 1055.50 | 1056.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1055.80 | 1056.65 | 1056.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 1058.90 | 1056.65 | 1056.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1060.50 | 1057.42 | 1057.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1060.50 | 1057.42 | 1057.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 1069.50 | 1060.49 | 1058.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 1079.00 | 1080.72 | 1074.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 10:00:00 | 1079.00 | 1080.72 | 1074.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1072.20 | 1078.28 | 1074.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:45:00 | 1072.40 | 1078.28 | 1074.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1084.00 | 1079.42 | 1075.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:15:00 | 1086.90 | 1079.42 | 1075.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1049.80 | 1082.22 | 1082.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 1049.80 | 1082.22 | 1082.35 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1064.00 | 1056.28 | 1056.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 12:15:00 | 1068.60 | 1058.74 | 1057.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1051.30 | 1061.25 | 1059.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1051.30 | 1061.25 | 1059.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1051.30 | 1061.25 | 1059.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:00:00 | 1051.30 | 1061.25 | 1059.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1052.20 | 1059.44 | 1058.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1051.50 | 1059.44 | 1058.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1051.70 | 1057.89 | 1058.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1049.00 | 1056.11 | 1057.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1032.80 | 1032.12 | 1040.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1032.80 | 1032.12 | 1040.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1033.10 | 1033.39 | 1038.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 1031.10 | 1033.39 | 1038.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 1036.70 | 1030.01 | 1029.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 1036.70 | 1030.01 | 1029.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 1039.40 | 1031.89 | 1030.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1034.70 | 1037.76 | 1035.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 1034.70 | 1037.76 | 1035.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1034.70 | 1037.76 | 1035.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1034.70 | 1037.76 | 1035.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1034.80 | 1037.17 | 1035.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1040.00 | 1037.17 | 1035.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1049.10 | 1039.56 | 1036.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1058.20 | 1044.63 | 1043.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1080.20 | 1051.19 | 1048.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 1103.90 | 1114.58 | 1115.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 1103.90 | 1114.58 | 1115.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1097.00 | 1108.49 | 1111.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1102.90 | 1099.28 | 1104.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1102.90 | 1099.28 | 1104.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1102.90 | 1099.28 | 1104.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1102.90 | 1099.28 | 1104.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1106.00 | 1100.62 | 1104.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 1101.60 | 1100.62 | 1104.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1103.90 | 1101.28 | 1104.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 1107.30 | 1101.28 | 1104.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 1102.20 | 1101.46 | 1104.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 1105.20 | 1101.46 | 1104.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1092.20 | 1090.90 | 1095.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1081.10 | 1089.31 | 1093.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 1082.40 | 1088.64 | 1091.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 1082.00 | 1087.59 | 1090.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1099.60 | 1090.67 | 1090.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1099.60 | 1090.67 | 1090.24 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 1079.80 | 1090.40 | 1090.80 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1097.00 | 1090.79 | 1090.59 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1082.70 | 1089.84 | 1090.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1070.90 | 1081.57 | 1085.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1115.90 | 1086.35 | 1086.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1115.90 | 1086.35 | 1086.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1115.90 | 1086.35 | 1086.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1112.00 | 1086.35 | 1086.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1113.00 | 1091.68 | 1089.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 1122.00 | 1104.76 | 1096.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1126.40 | 1127.52 | 1119.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 1126.40 | 1127.52 | 1119.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1122.30 | 1128.85 | 1122.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 1122.30 | 1128.85 | 1122.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1118.00 | 1126.68 | 1122.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1118.00 | 1126.68 | 1122.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1112.10 | 1123.76 | 1121.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 1112.90 | 1123.76 | 1121.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1122.10 | 1121.76 | 1120.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1126.70 | 1121.76 | 1120.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:45:00 | 1125.70 | 1123.60 | 1121.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1117.30 | 1122.34 | 1121.50 | SL hit (close<static) qty=1.00 sl=1120.00 alert=retest2 |

### Cycle 194 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 1114.40 | 1120.75 | 1120.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 1111.10 | 1118.82 | 1119.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1108.30 | 1107.94 | 1112.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 1108.30 | 1107.94 | 1112.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1098.70 | 1103.83 | 1108.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 1107.10 | 1103.83 | 1108.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1100.50 | 1103.06 | 1107.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 1109.60 | 1103.06 | 1107.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1107.90 | 1103.57 | 1106.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 1103.50 | 1104.72 | 1106.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:30:00 | 1103.80 | 1105.02 | 1106.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 1102.80 | 1105.02 | 1106.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1108.60 | 1106.35 | 1106.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1108.60 | 1106.35 | 1106.14 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1097.50 | 1104.38 | 1105.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 1088.30 | 1098.50 | 1101.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 1091.90 | 1090.97 | 1095.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1091.90 | 1090.97 | 1095.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1091.90 | 1090.97 | 1095.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 1095.00 | 1090.97 | 1095.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1094.20 | 1091.62 | 1094.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1094.20 | 1091.62 | 1094.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1093.00 | 1091.90 | 1094.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1095.00 | 1091.90 | 1094.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1095.00 | 1092.52 | 1094.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1096.10 | 1092.52 | 1094.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1089.00 | 1091.81 | 1094.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 1087.10 | 1090.37 | 1093.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 1087.90 | 1089.88 | 1092.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 1087.90 | 1089.04 | 1092.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1097.80 | 1090.79 | 1092.75 | SL hit (close>static) qty=1.00 sl=1096.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1109.90 | 1096.65 | 1095.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1116.80 | 1100.68 | 1097.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1097.80 | 1106.24 | 1102.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1097.80 | 1106.24 | 1102.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1097.80 | 1106.24 | 1102.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1097.80 | 1106.24 | 1102.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1103.20 | 1105.63 | 1102.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1105.70 | 1105.63 | 1102.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1105.10 | 1105.74 | 1103.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 1105.00 | 1104.96 | 1103.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 1105.00 | 1104.96 | 1103.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1105.00 | 1104.97 | 1103.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1116.00 | 1104.97 | 1103.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 1135.00 | 1139.18 | 1139.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1135.00 | 1139.18 | 1139.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1131.70 | 1137.68 | 1138.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1138.00 | 1129.79 | 1133.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1138.00 | 1129.79 | 1133.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1138.00 | 1129.79 | 1133.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1138.00 | 1129.79 | 1133.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1135.40 | 1130.92 | 1133.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1135.60 | 1130.92 | 1133.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1142.80 | 1136.03 | 1135.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 1154.70 | 1141.68 | 1138.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 1236.90 | 1238.84 | 1229.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 1231.40 | 1238.84 | 1229.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1239.00 | 1238.87 | 1230.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:00:00 | 1242.50 | 1238.61 | 1231.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 1223.00 | 1234.33 | 1232.22 | SL hit (close<static) qty=1.00 sl=1228.70 alert=retest2 |

### Cycle 200 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 1224.30 | 1230.61 | 1230.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 1217.50 | 1227.98 | 1229.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 1216.20 | 1215.00 | 1220.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:00:00 | 1216.20 | 1215.00 | 1220.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1212.70 | 1204.80 | 1210.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1212.70 | 1204.80 | 1210.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1209.20 | 1205.68 | 1210.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1205.00 | 1208.88 | 1210.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:45:00 | 1202.10 | 1207.27 | 1209.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 1216.00 | 1209.01 | 1210.00 | SL hit (close>static) qty=1.00 sl=1212.90 alert=retest2 |

### Cycle 201 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1218.60 | 1210.93 | 1210.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1220.50 | 1213.77 | 1212.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1228.70 | 1230.65 | 1224.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 1229.00 | 1230.65 | 1224.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 1227.90 | 1230.20 | 1225.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:45:00 | 1238.20 | 1230.58 | 1226.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 1231.60 | 1230.29 | 1226.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 15:15:00 | 1219.10 | 1228.05 | 1226.06 | SL hit (close<static) qty=1.00 sl=1224.30 alert=retest2 |

### Cycle 202 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1214.40 | 1222.72 | 1223.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1210.40 | 1220.26 | 1222.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1218.30 | 1216.64 | 1220.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1218.30 | 1216.64 | 1220.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1208.10 | 1214.38 | 1218.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1204.60 | 1210.82 | 1214.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1221.30 | 1214.22 | 1213.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1221.30 | 1214.22 | 1213.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1222.50 | 1215.87 | 1214.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1217.50 | 1217.91 | 1216.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1217.50 | 1217.91 | 1216.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1217.50 | 1217.91 | 1216.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1217.50 | 1217.91 | 1216.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1222.90 | 1218.91 | 1216.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1215.20 | 1218.91 | 1216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1216.40 | 1218.87 | 1217.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1216.40 | 1218.87 | 1217.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1217.40 | 1218.58 | 1217.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1214.40 | 1218.58 | 1217.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1217.90 | 1218.44 | 1217.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1213.80 | 1218.44 | 1217.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1204.00 | 1215.55 | 1216.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1194.00 | 1209.71 | 1213.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1182.50 | 1173.92 | 1185.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:00:00 | 1182.50 | 1173.92 | 1185.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1176.60 | 1173.62 | 1178.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1182.30 | 1173.62 | 1178.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1175.70 | 1174.04 | 1178.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:45:00 | 1174.10 | 1173.41 | 1177.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 1181.30 | 1174.79 | 1176.96 | SL hit (close>static) qty=1.00 sl=1180.70 alert=retest2 |

### Cycle 205 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1191.80 | 1179.11 | 1178.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 1194.50 | 1182.18 | 1180.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1183.50 | 1187.84 | 1183.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1183.50 | 1187.84 | 1183.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1183.50 | 1187.84 | 1183.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1186.50 | 1187.84 | 1183.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1186.00 | 1187.47 | 1184.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 1184.20 | 1187.47 | 1184.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1183.80 | 1186.73 | 1184.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 1183.80 | 1186.73 | 1184.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1185.00 | 1186.39 | 1184.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:30:00 | 1188.50 | 1186.61 | 1184.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 1187.50 | 1186.79 | 1184.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 1183.40 | 1186.11 | 1184.66 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |

### Cycle 206 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1180.80 | 1183.22 | 1183.49 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 12:15:00 | 1188.00 | 1183.42 | 1182.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 13:15:00 | 1192.90 | 1185.31 | 1183.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1225.00 | 1226.29 | 1217.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 1225.00 | 1226.29 | 1217.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1212.70 | 1223.80 | 1219.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1212.70 | 1223.80 | 1219.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1212.40 | 1221.52 | 1219.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 1212.40 | 1221.52 | 1219.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 1216.60 | 1220.20 | 1218.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1224.90 | 1220.20 | 1218.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 1219.70 | 1220.61 | 1219.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1213.20 | 1218.46 | 1218.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1213.20 | 1218.46 | 1218.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 1210.30 | 1215.81 | 1217.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1189.20 | 1187.10 | 1193.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:45:00 | 1191.60 | 1187.10 | 1193.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1180.80 | 1185.74 | 1191.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1180.80 | 1185.74 | 1191.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1173.40 | 1183.54 | 1189.48 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1204.80 | 1190.48 | 1189.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1208.60 | 1194.10 | 1191.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1202.40 | 1209.31 | 1204.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 1202.40 | 1209.31 | 1204.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1202.40 | 1209.31 | 1204.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1202.40 | 1209.31 | 1204.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1207.20 | 1208.89 | 1204.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1213.20 | 1208.01 | 1204.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 1213.80 | 1225.71 | 1225.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1213.80 | 1225.71 | 1225.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1206.00 | 1221.76 | 1223.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1169.80 | 1169.50 | 1181.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1169.80 | 1169.50 | 1181.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1178.70 | 1170.48 | 1178.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 1175.80 | 1170.48 | 1178.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1186.50 | 1173.69 | 1179.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 1186.50 | 1173.69 | 1179.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1183.00 | 1175.55 | 1179.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1187.50 | 1175.55 | 1179.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1180.70 | 1177.10 | 1179.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 1180.90 | 1177.10 | 1179.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1180.00 | 1177.68 | 1179.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1180.30 | 1177.68 | 1179.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1188.70 | 1179.88 | 1180.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1188.70 | 1179.88 | 1180.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1186.10 | 1181.13 | 1181.18 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1183.00 | 1181.50 | 1181.34 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 1176.50 | 1180.50 | 1180.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1168.60 | 1178.12 | 1179.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 12:15:00 | 1172.40 | 1171.43 | 1175.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 12:15:00 | 1172.40 | 1171.43 | 1175.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 1172.40 | 1171.43 | 1175.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 1175.10 | 1171.43 | 1175.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1165.00 | 1170.15 | 1174.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 1164.10 | 1169.88 | 1173.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 1142.60 | 1134.20 | 1133.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 1142.60 | 1134.20 | 1133.72 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1129.10 | 1133.18 | 1133.30 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1138.20 | 1134.18 | 1133.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 1139.10 | 1135.17 | 1134.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 1183.00 | 1186.23 | 1173.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 1183.00 | 1186.23 | 1173.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1174.50 | 1183.89 | 1173.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 1174.00 | 1183.89 | 1173.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1164.80 | 1180.07 | 1172.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 1178.70 | 1179.98 | 1173.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 1157.90 | 1175.56 | 1171.92 | SL hit (close<static) qty=1.00 sl=1163.10 alert=retest2 |

### Cycle 216 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 1158.60 | 1169.26 | 1169.51 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1178.00 | 1170.07 | 1169.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1228.60 | 1181.77 | 1175.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1180.40 | 1212.68 | 1200.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 1180.40 | 1212.68 | 1200.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1180.40 | 1212.68 | 1200.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 1182.80 | 1212.68 | 1200.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1209.10 | 1211.96 | 1201.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 1211.50 | 1211.73 | 1202.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:00:00 | 1212.10 | 1211.80 | 1203.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 1214.70 | 1212.38 | 1204.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 1183.30 | 1199.22 | 1200.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 1183.30 | 1199.22 | 1200.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 14:15:00 | 1180.30 | 1192.51 | 1196.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1184.90 | 1179.86 | 1187.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 1184.90 | 1179.86 | 1187.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1184.90 | 1179.86 | 1187.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 1184.90 | 1179.86 | 1187.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1191.20 | 1182.13 | 1187.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1191.20 | 1182.13 | 1187.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1195.00 | 1184.70 | 1188.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1185.20 | 1185.62 | 1188.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1195.60 | 1187.62 | 1188.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 1195.60 | 1187.62 | 1188.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1203.30 | 1190.75 | 1190.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1206.00 | 1193.80 | 1191.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 1197.80 | 1198.14 | 1194.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 1197.80 | 1198.14 | 1194.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1197.80 | 1198.14 | 1194.77 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 1146.80 | 1184.24 | 1189.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 1125.00 | 1172.39 | 1183.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 1145.50 | 1144.77 | 1154.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:30:00 | 1146.80 | 1144.77 | 1154.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1156.00 | 1147.66 | 1154.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1156.00 | 1147.66 | 1154.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1157.60 | 1149.65 | 1154.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1151.60 | 1149.65 | 1154.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 1171.70 | 1155.06 | 1153.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1171.70 | 1155.06 | 1153.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 1177.70 | 1159.59 | 1155.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 1182.70 | 1184.45 | 1174.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 1182.70 | 1184.45 | 1174.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1182.70 | 1184.45 | 1174.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 1178.10 | 1184.45 | 1174.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1189.80 | 1185.52 | 1175.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:45:00 | 1182.60 | 1185.52 | 1175.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1181.00 | 1184.62 | 1176.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1192.70 | 1184.62 | 1176.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1169.40 | 1181.57 | 1175.76 | SL hit (close<static) qty=1.00 sl=1172.90 alert=retest2 |

### Cycle 222 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 1155.20 | 1170.27 | 1171.26 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 1185.80 | 1171.86 | 1170.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 12:15:00 | 1199.70 | 1180.11 | 1174.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1173.00 | 1180.86 | 1176.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1173.00 | 1180.86 | 1176.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1173.00 | 1180.86 | 1176.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1173.00 | 1180.86 | 1176.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1172.00 | 1179.09 | 1175.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1162.80 | 1179.09 | 1175.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1162.60 | 1175.79 | 1174.69 | EMA400 retest candle locked (from upside) |

### Cycle 224 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 1163.10 | 1173.25 | 1173.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 1161.30 | 1170.86 | 1172.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1164.80 | 1162.89 | 1167.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1164.80 | 1162.89 | 1167.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1164.80 | 1162.89 | 1167.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1162.90 | 1162.89 | 1167.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1166.70 | 1163.65 | 1167.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1160.80 | 1163.65 | 1167.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:15:00 | 1162.20 | 1159.12 | 1160.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1181.80 | 1165.31 | 1163.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1181.80 | 1165.31 | 1163.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1190.80 | 1170.41 | 1165.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 1219.80 | 1225.02 | 1213.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 1219.80 | 1225.02 | 1213.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1217.80 | 1222.77 | 1214.79 | EMA400 retest candle locked (from upside) |

### Cycle 226 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1183.90 | 1206.17 | 1208.96 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 1217.10 | 1206.44 | 1205.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1222.20 | 1211.36 | 1207.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1216.30 | 1224.01 | 1218.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1216.30 | 1224.01 | 1218.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1216.30 | 1224.01 | 1218.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 10:45:00 | 1219.90 | 1223.57 | 1218.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 1281.00 | 1293.92 | 1294.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 1281.00 | 1293.92 | 1294.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 1257.60 | 1270.66 | 1277.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1270.50 | 1259.16 | 1266.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1270.50 | 1259.16 | 1266.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1270.50 | 1259.16 | 1266.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 1279.00 | 1259.16 | 1266.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1277.70 | 1262.87 | 1267.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1274.90 | 1262.87 | 1267.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1283.00 | 1266.90 | 1269.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 1284.20 | 1266.90 | 1269.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 1287.50 | 1271.02 | 1270.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 1292.30 | 1275.27 | 1272.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1274.00 | 1279.75 | 1275.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1274.00 | 1279.75 | 1275.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1274.00 | 1279.75 | 1275.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1274.00 | 1279.75 | 1275.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 1279.00 | 1279.60 | 1276.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 1270.80 | 1279.60 | 1276.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 1275.30 | 1278.74 | 1276.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 14:30:00 | 1283.00 | 1278.07 | 1276.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 15:15:00 | 1284.00 | 1278.07 | 1276.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1265.20 | 1276.44 | 1275.93 | SL hit (close<static) qty=1.00 sl=1268.30 alert=retest2 |

### Cycle 230 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1258.90 | 1272.94 | 1274.38 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1282.20 | 1276.28 | 1275.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1310.80 | 1283.75 | 1279.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 1305.10 | 1312.26 | 1305.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1305.10 | 1312.26 | 1305.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1305.10 | 1312.26 | 1305.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:00:00 | 1318.20 | 1313.42 | 1307.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1356.30 | 1309.52 | 1307.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 1273.70 | 1323.55 | 1320.08 | SL hit (close<static) qty=1.00 sl=1300.00 alert=retest2 |

### Cycle 232 — SELL (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 10:15:00 | 1327.30 | 1329.61 | 1329.79 | EMA200 below EMA400 |

### Cycle 233 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 1337.60 | 1331.21 | 1330.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 13:15:00 | 1341.20 | 1334.47 | 1332.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1328.40 | 1333.56 | 1332.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 1328.40 | 1333.56 | 1332.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1328.40 | 1333.56 | 1332.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 1324.10 | 1333.56 | 1332.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1327.10 | 1332.26 | 1331.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 1328.10 | 1332.26 | 1331.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 1343.60 | 1334.53 | 1332.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1346.80 | 1337.57 | 1335.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 1349.70 | 1342.25 | 1338.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 1346.40 | 1347.22 | 1342.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1348.30 | 1346.59 | 1342.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 1344.50 | 1346.17 | 1342.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:30:00 | 1343.60 | 1346.17 | 1342.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 1347.00 | 1346.34 | 1343.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:30:00 | 1345.40 | 1346.34 | 1343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1340.30 | 1345.56 | 1343.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 1340.30 | 1345.56 | 1343.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1340.00 | 1344.45 | 1343.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1353.90 | 1344.45 | 1343.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1365.40 | 1373.20 | 1373.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 234 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1365.40 | 1373.20 | 1373.85 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1379.60 | 1374.48 | 1374.38 | EMA200 above EMA400 |

### Cycle 236 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 1371.50 | 1373.88 | 1374.11 | EMA200 below EMA400 |

### Cycle 237 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1399.10 | 1378.64 | 1376.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 1404.20 | 1390.48 | 1383.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 1428.60 | 1431.13 | 1418.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:30:00 | 1428.00 | 1431.13 | 1418.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1422.90 | 1429.49 | 1418.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 1422.90 | 1429.49 | 1418.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1419.80 | 1427.55 | 1418.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 1419.80 | 1427.55 | 1418.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1420.60 | 1426.16 | 1419.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 1420.60 | 1426.16 | 1419.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1413.40 | 1423.61 | 1418.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1413.40 | 1423.61 | 1418.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1415.90 | 1422.07 | 1418.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1409.30 | 1422.07 | 1418.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1419.10 | 1421.47 | 1418.34 | EMA400 retest candle locked (from upside) |

### Cycle 238 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 1412.20 | 1418.63 | 1419.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1406.50 | 1416.21 | 1418.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 1394.20 | 1384.44 | 1390.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 1394.20 | 1384.44 | 1390.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1394.20 | 1384.44 | 1390.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 1394.20 | 1384.44 | 1390.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1397.00 | 1386.96 | 1391.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 1404.70 | 1386.96 | 1391.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 239 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1422.70 | 1397.36 | 1395.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1429.60 | 1403.81 | 1398.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1486.40 | 1489.05 | 1471.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 1486.40 | 1489.05 | 1471.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 11:45:00 | 601.05 | 2023-05-22 15:15:00 | 607.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-05-22 12:30:00 | 601.15 | 2023-05-22 15:15:00 | 607.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-05-26 09:15:00 | 598.45 | 2023-05-26 13:15:00 | 604.15 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-05-26 10:30:00 | 599.00 | 2023-05-26 13:15:00 | 604.15 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-06-06 14:00:00 | 655.45 | 2023-06-12 12:15:00 | 665.45 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2023-06-14 12:45:00 | 676.50 | 2023-06-14 14:15:00 | 669.65 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-06-15 09:15:00 | 681.15 | 2023-06-20 12:15:00 | 678.20 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-07-17 12:30:00 | 740.90 | 2023-07-18 10:15:00 | 733.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-07-17 13:30:00 | 739.15 | 2023-07-18 10:15:00 | 733.85 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-07-18 09:15:00 | 742.20 | 2023-07-18 10:15:00 | 733.85 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-07-18 09:45:00 | 740.45 | 2023-07-18 10:15:00 | 733.85 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-08-01 09:30:00 | 829.15 | 2023-08-01 12:15:00 | 819.70 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-08-16 11:15:00 | 875.70 | 2023-08-17 14:15:00 | 865.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-08-16 12:00:00 | 877.50 | 2023-08-17 14:15:00 | 865.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-08-17 09:30:00 | 878.15 | 2023-08-17 14:15:00 | 865.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-08-17 11:45:00 | 876.65 | 2023-08-17 14:15:00 | 865.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-08-24 13:15:00 | 837.30 | 2023-08-30 11:15:00 | 837.60 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2023-09-05 11:15:00 | 832.40 | 2023-09-05 12:15:00 | 834.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2023-09-08 13:30:00 | 858.55 | 2023-09-20 13:15:00 | 890.15 | STOP_HIT | 1.00 | 3.68% |
| BUY | retest2 | 2023-09-11 09:15:00 | 864.60 | 2023-09-20 13:15:00 | 890.15 | STOP_HIT | 1.00 | 2.96% |
| SELL | retest2 | 2023-09-26 12:15:00 | 869.35 | 2023-09-27 11:15:00 | 875.85 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-09-29 09:15:00 | 920.00 | 2023-10-04 11:15:00 | 880.50 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2023-10-12 09:15:00 | 914.00 | 2023-10-16 15:15:00 | 908.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-10-16 14:15:00 | 908.85 | 2023-10-16 15:15:00 | 908.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2023-10-16 15:00:00 | 908.90 | 2023-10-16 15:15:00 | 908.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2023-10-23 15:15:00 | 871.70 | 2023-10-31 09:15:00 | 863.15 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2023-10-25 09:45:00 | 872.55 | 2023-10-31 09:15:00 | 863.15 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2023-10-25 10:15:00 | 873.40 | 2023-10-31 09:15:00 | 863.15 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2023-10-25 11:00:00 | 874.00 | 2023-10-31 09:15:00 | 863.15 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2023-11-17 09:15:00 | 981.30 | 2023-11-28 10:15:00 | 1016.85 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2023-11-17 10:45:00 | 983.05 | 2023-11-28 10:15:00 | 1016.85 | STOP_HIT | 1.00 | 3.44% |
| SELL | retest2 | 2023-12-06 10:30:00 | 1019.60 | 2023-12-13 09:15:00 | 1041.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2023-12-07 14:00:00 | 1021.50 | 2023-12-13 09:15:00 | 1041.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-12-08 10:30:00 | 1022.80 | 2023-12-13 09:15:00 | 1041.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-12-18 09:15:00 | 1037.40 | 2023-12-18 12:15:00 | 1026.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-12-18 10:00:00 | 1040.00 | 2023-12-18 12:15:00 | 1026.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-01-02 09:15:00 | 1104.35 | 2024-01-10 11:15:00 | 1096.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-01-03 10:00:00 | 1087.25 | 2024-01-10 11:15:00 | 1096.80 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-01-24 15:15:00 | 1165.00 | 2024-01-31 09:15:00 | 1146.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-01-25 10:00:00 | 1165.20 | 2024-01-31 12:15:00 | 1150.45 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-01-25 11:15:00 | 1164.15 | 2024-01-31 12:15:00 | 1150.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-01-29 11:45:00 | 1162.50 | 2024-01-31 12:15:00 | 1150.45 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-01-30 13:45:00 | 1171.25 | 2024-01-31 12:15:00 | 1150.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2024-02-05 09:15:00 | 1044.80 | 2024-02-05 14:15:00 | 1010.28 | PARTIAL | 0.50 | 3.30% |
| SELL | retest1 | 2024-02-05 09:15:00 | 1044.80 | 2024-02-06 09:15:00 | 1042.15 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-05 10:15:00 | 1063.45 | 2024-02-07 09:15:00 | 992.56 | PARTIAL | 0.50 | 6.67% |
| SELL | retest1 | 2024-02-05 10:15:00 | 1063.45 | 2024-02-07 15:15:00 | 1005.00 | STOP_HIT | 0.50 | 5.50% |
| SELL | retest2 | 2024-02-08 11:15:00 | 1013.50 | 2024-02-12 14:15:00 | 1017.75 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-02-08 12:15:00 | 1004.60 | 2024-02-12 14:15:00 | 1017.75 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-02-08 13:45:00 | 1012.75 | 2024-02-12 14:15:00 | 1017.75 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-02-08 14:15:00 | 1010.50 | 2024-02-12 14:15:00 | 1017.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-02-09 10:15:00 | 994.70 | 2024-02-12 14:15:00 | 1017.75 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-02-09 15:00:00 | 996.70 | 2024-02-12 14:15:00 | 1017.75 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-02-20 14:30:00 | 1050.10 | 2024-02-26 09:15:00 | 1025.30 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-02-22 11:00:00 | 1047.85 | 2024-02-26 09:15:00 | 1025.30 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-02-22 11:30:00 | 1046.00 | 2024-02-26 09:15:00 | 1025.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-02-22 13:00:00 | 1045.95 | 2024-02-26 09:15:00 | 1025.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-03-01 09:15:00 | 1016.00 | 2024-03-02 09:15:00 | 1067.00 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2024-03-01 11:15:00 | 1023.25 | 2024-03-02 09:15:00 | 1067.00 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-03-19 09:15:00 | 1010.70 | 2024-03-20 13:15:00 | 1012.95 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-03-26 11:15:00 | 1026.95 | 2024-04-01 09:15:00 | 1129.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-24 12:15:00 | 1084.95 | 2024-04-25 13:15:00 | 1101.75 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-04-24 14:45:00 | 1084.95 | 2024-04-25 13:15:00 | 1101.75 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-04-25 09:30:00 | 1082.50 | 2024-04-25 13:15:00 | 1101.75 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-04-25 10:45:00 | 1081.75 | 2024-04-25 13:15:00 | 1101.75 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-05-02 14:30:00 | 1161.95 | 2024-05-07 09:15:00 | 1137.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-05-03 09:15:00 | 1167.70 | 2024-05-07 09:15:00 | 1137.20 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-05-03 10:15:00 | 1161.25 | 2024-05-07 09:15:00 | 1137.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-05-03 10:45:00 | 1162.15 | 2024-05-07 09:15:00 | 1137.20 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-05-06 10:45:00 | 1160.00 | 2024-05-07 09:15:00 | 1137.20 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-05-06 11:45:00 | 1167.60 | 2024-05-07 09:15:00 | 1137.20 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-05-09 09:15:00 | 1127.45 | 2024-05-09 09:15:00 | 1146.50 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-05-09 12:45:00 | 1128.85 | 2024-05-13 10:15:00 | 1141.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-05-10 14:15:00 | 1128.90 | 2024-05-13 10:15:00 | 1141.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-05-18 09:15:00 | 1193.10 | 2024-05-27 13:15:00 | 1204.25 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2024-06-19 14:45:00 | 1225.15 | 2024-06-21 09:15:00 | 1255.10 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-07-12 14:30:00 | 1326.15 | 2024-07-18 12:15:00 | 1327.30 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-07-12 15:00:00 | 1327.00 | 2024-07-18 12:15:00 | 1327.30 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-07-25 11:00:00 | 1357.05 | 2024-08-05 13:15:00 | 1413.60 | STOP_HIT | 1.00 | 4.17% |
| BUY | retest2 | 2024-07-25 13:45:00 | 1359.80 | 2024-08-05 13:15:00 | 1413.60 | STOP_HIT | 1.00 | 3.96% |
| BUY | retest2 | 2024-08-09 12:45:00 | 1465.00 | 2024-08-09 15:15:00 | 1442.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-08-09 13:45:00 | 1464.95 | 2024-08-09 15:15:00 | 1442.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-08-12 09:30:00 | 1466.50 | 2024-08-12 10:15:00 | 1450.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-08-12 15:15:00 | 1465.55 | 2024-08-16 09:15:00 | 1427.40 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-08-30 09:45:00 | 1568.25 | 2024-09-02 13:15:00 | 1554.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-08-30 12:00:00 | 1568.20 | 2024-09-02 13:15:00 | 1554.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-08-30 15:15:00 | 1568.05 | 2024-09-02 13:15:00 | 1554.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-09-02 10:30:00 | 1571.95 | 2024-09-02 13:15:00 | 1554.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-09-03 13:45:00 | 1553.95 | 2024-09-12 09:15:00 | 1565.15 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-09-05 10:45:00 | 1548.75 | 2024-09-12 09:15:00 | 1565.15 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-09-05 12:00:00 | 1553.25 | 2024-09-12 09:15:00 | 1565.15 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-09-19 10:15:00 | 1536.60 | 2024-09-25 11:15:00 | 1511.90 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2024-10-16 11:15:00 | 1467.00 | 2024-10-21 09:15:00 | 1491.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-10-17 15:15:00 | 1463.85 | 2024-10-21 09:15:00 | 1491.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-10-18 11:15:00 | 1465.15 | 2024-10-21 09:15:00 | 1491.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-10-23 14:15:00 | 1443.15 | 2024-11-05 10:15:00 | 1370.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:30:00 | 1447.25 | 2024-11-05 10:15:00 | 1374.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:15:00 | 1443.15 | 2024-11-05 14:15:00 | 1396.00 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2024-10-25 09:30:00 | 1447.25 | 2024-11-05 14:15:00 | 1396.00 | STOP_HIT | 0.50 | 3.54% |
| BUY | retest2 | 2024-12-03 15:00:00 | 1268.75 | 2024-12-04 09:15:00 | 1241.90 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-12-04 11:30:00 | 1258.25 | 2024-12-05 10:15:00 | 1239.15 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-04 12:15:00 | 1259.25 | 2024-12-05 10:15:00 | 1239.15 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-12-04 13:30:00 | 1257.15 | 2024-12-05 10:15:00 | 1239.15 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-12-23 09:45:00 | 1246.95 | 2024-12-26 12:15:00 | 1251.80 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-12-23 10:45:00 | 1253.50 | 2024-12-26 12:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-01-07 12:15:00 | 1317.25 | 2025-01-09 14:15:00 | 1251.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 12:15:00 | 1317.25 | 2025-01-10 13:15:00 | 1185.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 15:15:00 | 1060.30 | 2025-03-05 09:15:00 | 1077.60 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-03-10 09:45:00 | 1104.50 | 2025-03-10 15:15:00 | 1088.20 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-03-25 13:15:00 | 1191.25 | 2025-03-26 10:15:00 | 1181.15 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-03-26 09:45:00 | 1192.20 | 2025-03-26 10:15:00 | 1181.15 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-03-28 13:45:00 | 1153.35 | 2025-04-03 09:15:00 | 1204.95 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-03-28 15:15:00 | 1155.00 | 2025-04-03 09:15:00 | 1204.95 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-04-01 10:45:00 | 1152.30 | 2025-04-03 09:15:00 | 1204.95 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-04-01 12:00:00 | 1154.85 | 2025-04-03 09:15:00 | 1204.95 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2025-04-28 11:15:00 | 1244.10 | 2025-04-29 09:15:00 | 1208.80 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-04-28 15:00:00 | 1247.20 | 2025-04-29 09:15:00 | 1208.80 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-05-02 12:15:00 | 1207.70 | 2025-05-05 11:15:00 | 1249.60 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-05-02 13:00:00 | 1209.30 | 2025-05-05 11:15:00 | 1249.60 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-05-02 13:45:00 | 1211.50 | 2025-05-05 11:15:00 | 1249.60 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1210.10 | 2025-05-05 11:15:00 | 1249.60 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-05-14 14:45:00 | 1217.10 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-05-15 09:45:00 | 1219.00 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-15 11:15:00 | 1217.80 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-05-15 11:45:00 | 1218.30 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-05-22 11:30:00 | 1188.10 | 2025-06-02 09:15:00 | 1133.83 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-05-22 15:15:00 | 1187.10 | 2025-06-02 09:15:00 | 1133.92 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2025-05-23 12:45:00 | 1184.40 | 2025-06-02 09:15:00 | 1133.73 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-05-23 13:30:00 | 1185.90 | 2025-06-02 09:15:00 | 1133.83 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2025-05-22 11:30:00 | 1188.10 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-05-22 15:15:00 | 1187.10 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-05-23 12:45:00 | 1184.40 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-05-23 13:30:00 | 1185.90 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2025-05-26 10:15:00 | 1193.50 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-05-26 11:30:00 | 1193.60 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.09% |
| SELL | retest2 | 2025-05-27 10:00:00 | 1193.40 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-05-27 10:30:00 | 1193.50 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-05-28 09:15:00 | 1167.50 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-06-06 14:00:00 | 1163.50 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-06-06 15:15:00 | 1161.10 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-06-10 09:45:00 | 1162.70 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-06-10 11:00:00 | 1164.90 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-06-17 10:00:00 | 1134.50 | 2025-06-24 09:15:00 | 1112.30 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1153.60 | 2025-07-01 09:15:00 | 1119.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-07-24 10:15:00 | 1112.00 | 2025-07-25 14:15:00 | 1125.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-24 14:15:00 | 1115.90 | 2025-07-25 14:15:00 | 1125.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-25 12:00:00 | 1112.50 | 2025-07-25 14:15:00 | 1125.40 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-29 11:15:00 | 1146.50 | 2025-08-01 09:15:00 | 1103.20 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-08-05 11:30:00 | 1066.40 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-08-05 14:15:00 | 1066.40 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-08-07 10:15:00 | 1066.30 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1061.10 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-08-18 13:15:00 | 1086.90 | 2025-08-20 09:15:00 | 1049.80 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-08-29 14:15:00 | 1031.10 | 2025-09-03 12:15:00 | 1036.70 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1058.20 | 2025-09-23 12:15:00 | 1103.90 | STOP_HIT | 1.00 | 4.32% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1080.20 | 2025-09-23 12:15:00 | 1103.90 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1081.10 | 2025-10-03 09:15:00 | 1099.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-09-30 12:15:00 | 1082.40 | 2025-10-03 09:15:00 | 1099.60 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-30 13:15:00 | 1082.00 | 2025-10-03 09:15:00 | 1099.60 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1126.70 | 2025-10-15 11:15:00 | 1117.30 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-15 10:45:00 | 1125.70 | 2025-10-15 11:15:00 | 1117.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-10-20 12:45:00 | 1103.50 | 2025-10-23 11:15:00 | 1108.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-20 13:30:00 | 1103.80 | 2025-10-23 11:15:00 | 1108.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-20 14:15:00 | 1102.80 | 2025-10-23 11:15:00 | 1108.60 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-10-28 11:30:00 | 1087.10 | 2025-10-28 14:15:00 | 1097.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-10-28 13:00:00 | 1087.90 | 2025-10-28 14:15:00 | 1097.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-28 13:30:00 | 1087.90 | 2025-10-28 14:15:00 | 1097.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1105.70 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.65% |
| BUY | retest2 | 2025-10-30 12:30:00 | 1105.10 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-10-30 14:30:00 | 1105.00 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-10-30 15:15:00 | 1105.00 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1116.00 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2025-11-19 12:00:00 | 1242.50 | 2025-11-20 09:15:00 | 1223.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1205.00 | 2025-11-26 10:15:00 | 1216.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-11-26 09:45:00 | 1202.10 | 2025-11-26 10:15:00 | 1216.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-28 13:45:00 | 1238.20 | 2025-11-28 15:15:00 | 1219.10 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-28 14:30:00 | 1231.60 | 2025-11-28 15:15:00 | 1219.10 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-12-03 09:30:00 | 1204.60 | 2025-12-04 12:15:00 | 1221.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-11 12:45:00 | 1174.10 | 2025-12-12 09:15:00 | 1181.30 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-15 13:30:00 | 1188.50 | 2025-12-15 15:15:00 | 1183.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-15 15:00:00 | 1187.50 | 2025-12-15 15:15:00 | 1183.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1224.90 | 2025-12-24 13:15:00 | 1213.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-24 12:00:00 | 1219.70 | 2025-12-24 13:15:00 | 1213.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1213.20 | 2026-01-08 13:15:00 | 1213.80 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1164.10 | 2026-01-28 15:15:00 | 1142.60 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2026-02-02 09:45:00 | 1178.70 | 2026-02-02 10:15:00 | 1157.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-04 13:15:00 | 1211.50 | 2026-02-05 11:15:00 | 1183.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-04 14:00:00 | 1212.10 | 2026-02-05 11:15:00 | 1183.30 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-04 15:00:00 | 1214.70 | 2026-02-05 11:15:00 | 1183.30 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1151.60 | 2026-02-16 10:15:00 | 1171.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-18 09:15:00 | 1192.70 | 2026-02-18 09:15:00 | 1169.40 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1160.80 | 2026-02-25 09:15:00 | 1181.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-24 14:15:00 | 1162.20 | 2026-02-25 09:15:00 | 1181.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-03-09 10:45:00 | 1219.90 | 2026-03-16 10:15:00 | 1281.00 | STOP_HIT | 1.00 | 5.01% |
| BUY | retest2 | 2026-03-23 14:30:00 | 1283.00 | 2026-03-24 09:15:00 | 1265.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-03-23 15:15:00 | 1284.00 | 2026-03-24 09:15:00 | 1265.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-03-30 12:00:00 | 1318.20 | 2026-04-02 09:15:00 | 1273.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1356.30 | 2026-04-02 09:15:00 | 1273.70 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest2 | 2026-04-02 11:30:00 | 1317.10 | 2026-04-08 10:15:00 | 1327.30 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1346.80 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2026-04-10 11:30:00 | 1349.70 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2026-04-13 09:45:00 | 1346.40 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1348.30 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1353.90 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.85% |
