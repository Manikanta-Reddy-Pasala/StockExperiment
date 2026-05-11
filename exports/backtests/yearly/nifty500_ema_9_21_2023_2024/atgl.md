# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 632.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 190 |
| ALERT1 | 121 |
| ALERT2 | 120 |
| ALERT2_SKIP | 57 |
| ALERT3 | 305 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 190 |
| PARTIAL | 41 |
| TARGET_HIT | 29 |
| STOP_HIT | 171 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 239 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 116 / 123
- **Target hits / Stop hits / Partials:** 28 / 171 / 40
- **Avg / median % per leg:** 1.87% / -0.15%
- **Sum % (uncompounded):** 447.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 13 | 22.4% | 6 | 50 | 2 | 0.37% | 21.5% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 0 | 5 | 2 | 2.70% | 18.9% |
| BUY @ 3rd Alert (retest2) | 51 | 7 | 13.7% | 6 | 45 | 0 | 0.05% | 2.6% |
| SELL (all) | 181 | 103 | 56.9% | 22 | 121 | 38 | 2.35% | 425.9% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | 2.26% | 13.6% |
| SELL @ 3rd Alert (retest2) | 175 | 99 | 56.6% | 22 | 117 | 36 | 2.36% | 412.3% |
| retest1 (combined) | 13 | 10 | 76.9% | 0 | 9 | 4 | 2.50% | 32.4% |
| retest2 (combined) | 226 | 106 | 46.9% | 28 | 162 | 36 | 1.84% | 414.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 722.50 | 692.99 | 690.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 758.60 | 720.05 | 706.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 783.35 | 784.39 | 762.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 09:45:00 | 774.25 | 784.39 | 762.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 777.40 | 785.86 | 773.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 777.00 | 785.86 | 773.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 773.95 | 783.48 | 773.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:45:00 | 772.55 | 783.48 | 773.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 770.25 | 780.83 | 773.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:30:00 | 773.30 | 780.83 | 773.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 12:15:00 | 766.00 | 777.87 | 772.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 12:45:00 | 764.10 | 777.87 | 772.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 13:15:00 | 771.55 | 776.60 | 772.77 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 09:15:00 | 740.10 | 764.78 | 767.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 12:15:00 | 731.10 | 750.61 | 760.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 693.50 | 681.55 | 700.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-01 09:45:00 | 697.35 | 681.55 | 700.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 672.05 | 687.46 | 695.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 09:15:00 | 667.00 | 671.01 | 672.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 11:30:00 | 666.65 | 665.09 | 665.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 12:15:00 | 669.80 | 666.04 | 665.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 12:15:00 | 669.80 | 666.04 | 665.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 11:15:00 | 673.00 | 669.62 | 667.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 09:15:00 | 667.70 | 670.06 | 668.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 09:15:00 | 667.70 | 670.06 | 668.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 667.70 | 670.06 | 668.90 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 12:15:00 | 664.20 | 667.49 | 667.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 13:15:00 | 659.50 | 665.89 | 667.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 660.30 | 657.62 | 659.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 660.30 | 657.62 | 659.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 660.30 | 657.62 | 659.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 660.30 | 657.62 | 659.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 659.55 | 658.01 | 659.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:30:00 | 657.60 | 657.27 | 659.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:15:00 | 624.72 | 639.44 | 646.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 640.00 | 639.44 | 646.07 | SL hit (close>static) qty=0.50 sl=639.44 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 656.55 | 647.02 | 645.82 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 15:15:00 | 650.50 | 653.29 | 653.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 10:15:00 | 647.90 | 651.68 | 652.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 15:15:00 | 651.10 | 650.05 | 651.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 15:15:00 | 651.10 | 650.05 | 651.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 651.10 | 650.05 | 651.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 651.60 | 650.05 | 651.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 649.05 | 649.85 | 651.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 09:15:00 | 645.00 | 649.36 | 650.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 14:30:00 | 643.80 | 640.49 | 643.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 09:15:00 | 647.55 | 635.17 | 634.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 647.55 | 635.17 | 634.54 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 09:15:00 | 639.45 | 642.70 | 642.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 11:15:00 | 638.20 | 639.69 | 640.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 640.85 | 635.59 | 637.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 640.85 | 635.59 | 637.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 640.85 | 635.59 | 637.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 640.85 | 635.59 | 637.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 638.00 | 636.07 | 637.19 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 13:15:00 | 645.60 | 639.12 | 638.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 14:15:00 | 662.65 | 643.83 | 640.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 09:15:00 | 655.40 | 659.33 | 653.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:45:00 | 654.80 | 659.33 | 653.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 653.10 | 656.77 | 653.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 15:00:00 | 655.70 | 656.55 | 654.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 09:15:00 | 656.00 | 656.16 | 654.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 649.00 | 658.92 | 660.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 649.00 | 658.92 | 660.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 15:15:00 | 645.00 | 653.64 | 655.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 14:15:00 | 649.00 | 648.88 | 651.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 14:15:00 | 649.00 | 648.88 | 651.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 649.00 | 648.88 | 651.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 15:00:00 | 649.00 | 648.88 | 651.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 661.45 | 651.09 | 652.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:00:00 | 661.45 | 651.09 | 652.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 655.50 | 651.97 | 652.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:15:00 | 655.00 | 651.97 | 652.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 12:30:00 | 653.70 | 652.97 | 652.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 13:15:00 | 655.80 | 653.54 | 653.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 13:15:00 | 655.80 | 653.54 | 653.23 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 15:15:00 | 652.00 | 652.99 | 653.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 09:15:00 | 648.00 | 651.99 | 652.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 09:15:00 | 649.70 | 648.48 | 650.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 09:15:00 | 649.70 | 648.48 | 650.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 649.70 | 648.48 | 650.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 10:30:00 | 646.40 | 648.46 | 649.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 14:30:00 | 645.55 | 648.45 | 649.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 15:00:00 | 646.95 | 648.45 | 649.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 09:15:00 | 636.00 | 649.52 | 649.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 635.10 | 635.44 | 638.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 10:45:00 | 633.85 | 635.15 | 638.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 12:15:00 | 633.60 | 634.99 | 637.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 14:30:00 | 634.25 | 634.93 | 637.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 641.20 | 635.95 | 637.19 | SL hit (close>static) qty=1.00 sl=639.90 alert=retest2 |

### Cycle 13 — BUY (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 11:15:00 | 662.55 | 642.07 | 639.80 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 12:15:00 | 662.60 | 663.38 | 663.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 13:15:00 | 661.90 | 663.09 | 663.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 15:15:00 | 663.60 | 663.05 | 663.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 15:15:00 | 663.60 | 663.05 | 663.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 663.60 | 663.05 | 663.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 663.50 | 663.05 | 663.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 653.60 | 661.16 | 662.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 10:15:00 | 650.95 | 661.16 | 662.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 10:45:00 | 652.00 | 659.84 | 661.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 14:15:00 | 652.00 | 656.51 | 659.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 10:30:00 | 650.85 | 655.35 | 657.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 657.40 | 655.64 | 657.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:45:00 | 657.95 | 655.64 | 657.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 654.50 | 655.41 | 657.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-29 13:15:00 | 659.75 | 657.76 | 657.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 13:15:00 | 659.75 | 657.76 | 657.71 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 657.25 | 657.66 | 657.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 09:15:00 | 654.35 | 656.88 | 657.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 12:15:00 | 638.45 | 637.31 | 642.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 12:45:00 | 637.45 | 637.31 | 642.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 635.25 | 635.81 | 640.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 14:15:00 | 633.15 | 635.17 | 638.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 09:30:00 | 633.75 | 634.22 | 637.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 10:00:00 | 633.95 | 634.22 | 637.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 12:30:00 | 632.75 | 633.81 | 636.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 635.35 | 634.09 | 635.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-06 14:15:00 | 642.95 | 636.91 | 636.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 14:15:00 | 642.95 | 636.91 | 636.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 13:15:00 | 651.00 | 645.33 | 641.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 13:15:00 | 647.25 | 647.87 | 644.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-08 13:30:00 | 647.50 | 647.87 | 644.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 646.00 | 647.50 | 645.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 15:00:00 | 646.00 | 647.50 | 645.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 644.00 | 646.80 | 644.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:45:00 | 648.95 | 646.88 | 645.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 10:30:00 | 648.85 | 653.55 | 650.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 13:15:00 | 647.00 | 650.80 | 649.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 14:15:00 | 642.85 | 648.42 | 648.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 642.85 | 648.42 | 648.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 634.80 | 644.91 | 647.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 10:15:00 | 641.00 | 640.22 | 642.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-14 10:45:00 | 641.45 | 640.22 | 642.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 643.60 | 640.93 | 642.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:30:00 | 643.80 | 640.93 | 642.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 643.00 | 641.35 | 642.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 641.85 | 641.35 | 642.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 643.25 | 641.73 | 642.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 15:00:00 | 635.70 | 640.69 | 641.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 14:15:00 | 640.00 | 641.23 | 641.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 14:45:00 | 639.95 | 640.98 | 641.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 10:15:00 | 644.85 | 640.24 | 640.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 10:15:00 | 644.85 | 640.24 | 640.18 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 637.50 | 639.74 | 640.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 09:15:00 | 635.40 | 637.42 | 638.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 15:15:00 | 635.00 | 634.41 | 635.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 15:15:00 | 635.00 | 634.41 | 635.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 635.00 | 634.41 | 635.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 632.50 | 634.41 | 635.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 630.00 | 633.53 | 635.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 11:00:00 | 628.95 | 632.61 | 634.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 11:45:00 | 629.00 | 631.93 | 634.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 12:15:00 | 629.00 | 631.93 | 634.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 597.50 | 603.85 | 607.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 597.55 | 603.85 | 607.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 597.55 | 603.85 | 607.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-10 09:15:00 | 594.20 | 594.13 | 599.58 | SL hit (close>ema200) qty=0.50 sl=594.13 alert=retest2 |

### Cycle 21 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 615.60 | 601.41 | 600.25 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 12:15:00 | 602.40 | 605.51 | 605.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 14:15:00 | 602.00 | 604.80 | 605.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 599.60 | 598.10 | 600.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 599.60 | 598.10 | 600.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 599.60 | 598.10 | 600.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:45:00 | 600.30 | 598.10 | 600.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 598.60 | 598.20 | 600.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 600.40 | 598.20 | 600.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 596.30 | 596.91 | 598.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:00:00 | 594.55 | 596.44 | 598.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:45:00 | 594.20 | 595.90 | 597.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 12:30:00 | 594.50 | 595.46 | 597.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:45:00 | 594.40 | 591.30 | 593.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 592.50 | 591.54 | 593.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:30:00 | 593.00 | 591.54 | 593.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 591.75 | 591.58 | 593.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:45:00 | 590.90 | 591.54 | 592.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:15:00 | 591.20 | 591.54 | 592.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:45:00 | 591.05 | 591.32 | 592.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 15:15:00 | 589.40 | 591.44 | 592.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 580.70 | 588.97 | 591.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 10:15:00 | 578.50 | 588.97 | 591.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 564.82 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 564.49 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 564.77 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 564.68 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 561.35 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 561.64 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 561.50 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 559.93 | 569.39 | 577.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 14:15:00 | 549.57 | 563.66 | 573.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-26 09:15:00 | 535.10 | 553.88 | 567.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 23 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 564.50 | 554.24 | 554.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 10:15:00 | 570.15 | 557.42 | 555.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 15:15:00 | 562.70 | 562.82 | 559.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:15:00 | 570.75 | 562.82 | 559.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 565.40 | 568.07 | 564.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:00:00 | 565.40 | 568.07 | 564.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 564.85 | 567.42 | 564.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 09:15:00 | 567.00 | 567.42 | 564.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 10:15:00 | 557.85 | 565.12 | 563.71 | SL hit (close<ema400) qty=1.00 sl=563.71 alert=retest1 |

### Cycle 24 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 552.50 | 560.78 | 561.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 551.45 | 557.90 | 560.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 13:15:00 | 558.55 | 555.77 | 557.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 13:15:00 | 558.55 | 555.77 | 557.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 558.55 | 555.77 | 557.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:00:00 | 558.55 | 555.77 | 557.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 554.05 | 555.42 | 557.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:30:00 | 557.85 | 555.42 | 557.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 557.75 | 555.65 | 557.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:30:00 | 560.10 | 555.65 | 557.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 556.90 | 555.90 | 557.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 14:30:00 | 555.50 | 555.53 | 556.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 09:30:00 | 553.85 | 554.91 | 556.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 14:15:00 | 542.30 | 541.34 | 541.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 542.30 | 541.34 | 541.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 545.00 | 542.10 | 541.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 13:15:00 | 542.60 | 543.08 | 542.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 13:15:00 | 542.60 | 543.08 | 542.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 13:15:00 | 542.60 | 543.08 | 542.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 13:45:00 | 542.30 | 543.08 | 542.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 14:15:00 | 540.35 | 542.54 | 542.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 14:30:00 | 539.45 | 542.54 | 542.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 540.80 | 542.19 | 542.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:15:00 | 537.65 | 542.19 | 542.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 09:15:00 | 538.05 | 541.36 | 541.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 15:15:00 | 536.30 | 538.14 | 539.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 15:15:00 | 534.70 | 534.57 | 536.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-20 09:15:00 | 531.40 | 534.57 | 536.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 530.20 | 533.69 | 536.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 11:30:00 | 528.35 | 532.16 | 534.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 12:30:00 | 528.95 | 531.41 | 534.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 13:00:00 | 528.40 | 531.41 | 534.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 09:15:00 | 541.85 | 532.90 | 534.00 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 10:15:00 | 542.85 | 534.89 | 534.80 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 530.90 | 535.61 | 536.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 09:15:00 | 530.35 | 531.49 | 533.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 11:15:00 | 532.00 | 531.43 | 532.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 12:00:00 | 532.00 | 531.43 | 532.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 537.45 | 532.63 | 533.21 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 13:15:00 | 539.55 | 534.01 | 533.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 618.60 | 551.98 | 542.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 703.95 | 706.41 | 660.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 766.85 | 735.25 | 717.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 766.85 | 735.25 | 717.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 10:15:00 | 769.10 | 735.25 | 717.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-05 12:15:00 | 846.01 | 778.66 | 743.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 1040.00 | 1111.89 | 1117.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 10:15:00 | 1017.95 | 1093.10 | 1108.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 1040.25 | 1036.86 | 1067.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 1040.25 | 1036.86 | 1067.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1040.25 | 1036.86 | 1067.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:30:00 | 1063.45 | 1036.86 | 1067.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 1052.10 | 1045.74 | 1061.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 14:45:00 | 1058.00 | 1045.74 | 1061.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 1058.50 | 1049.77 | 1060.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 09:30:00 | 1033.90 | 1047.49 | 1054.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 1029.65 | 1045.27 | 1053.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 15:00:00 | 1029.75 | 1040.59 | 1048.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 13:30:00 | 1034.50 | 1032.89 | 1039.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 1025.10 | 1031.33 | 1038.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 14:45:00 | 1034.25 | 1031.33 | 1038.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 1035.45 | 1031.14 | 1037.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:30:00 | 1039.45 | 1031.14 | 1037.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 1032.95 | 1031.50 | 1036.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 11:15:00 | 1030.60 | 1031.50 | 1036.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-20 13:15:00 | 930.51 | 1016.26 | 1028.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 1002.60 | 996.49 | 996.08 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 09:15:00 | 986.75 | 995.60 | 996.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 980.15 | 990.82 | 993.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 990.50 | 987.35 | 991.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-29 11:00:00 | 990.50 | 987.35 | 991.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 990.25 | 987.93 | 990.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 13:15:00 | 986.80 | 988.22 | 990.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 14:45:00 | 986.00 | 987.77 | 990.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 15:15:00 | 986.80 | 987.77 | 990.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 09:30:00 | 987.00 | 987.43 | 989.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 1000.10 | 989.96 | 990.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 1000.10 | 989.96 | 990.51 | SL hit (close>static) qty=1.00 sl=995.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 11:15:00 | 998.65 | 991.70 | 991.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 13:15:00 | 1008.45 | 1001.60 | 997.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 14:15:00 | 999.70 | 1001.22 | 997.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 15:00:00 | 999.70 | 1001.22 | 997.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 1091.35 | 1097.39 | 1085.50 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 13:15:00 | 1070.95 | 1079.14 | 1080.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-11 11:15:00 | 1064.20 | 1068.73 | 1071.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 1066.90 | 1065.69 | 1068.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 1066.90 | 1065.69 | 1068.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 1066.90 | 1065.69 | 1068.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 10:00:00 | 1066.90 | 1065.69 | 1068.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 10:15:00 | 1063.90 | 1065.33 | 1068.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 12:30:00 | 1060.50 | 1063.99 | 1067.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 13:15:00 | 1060.00 | 1063.99 | 1067.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 14:30:00 | 1061.00 | 1062.57 | 1065.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 09:15:00 | 1007.47 | 1028.09 | 1039.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 09:15:00 | 1007.00 | 1028.09 | 1039.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 09:15:00 | 1007.95 | 1028.09 | 1039.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 999.75 | 992.43 | 1004.82 | SL hit (close>ema200) qty=0.50 sl=992.43 alert=retest2 |

### Cycle 35 — BUY (started 2024-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 14:15:00 | 1051.80 | 999.27 | 998.64 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 10:15:00 | 1000.00 | 1003.63 | 1004.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 997.10 | 1002.32 | 1003.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 15:15:00 | 1002.00 | 1001.65 | 1002.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 15:15:00 | 1002.00 | 1001.65 | 1002.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 1002.00 | 1001.65 | 1002.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 1009.25 | 1001.65 | 1002.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1006.95 | 1002.71 | 1003.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 1011.25 | 1002.71 | 1003.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 998.35 | 1001.84 | 1002.63 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 1039.80 | 1009.83 | 1005.90 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 15:15:00 | 1011.00 | 1020.13 | 1021.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 11:15:00 | 1004.50 | 1014.56 | 1018.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 13:15:00 | 1015.65 | 1013.62 | 1017.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 13:15:00 | 1015.65 | 1013.62 | 1017.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 1015.65 | 1013.62 | 1017.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:00:00 | 1015.65 | 1013.62 | 1017.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 1001.90 | 1011.28 | 1015.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:45:00 | 1014.40 | 1011.28 | 1015.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1008.05 | 1008.99 | 1013.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 14:15:00 | 1002.10 | 1008.32 | 1011.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 15:00:00 | 999.65 | 1006.59 | 1010.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 09:15:00 | 1049.60 | 1006.46 | 1003.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 1049.60 | 1006.46 | 1003.77 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 14:15:00 | 1019.45 | 1021.46 | 1021.54 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 1025.00 | 1022.17 | 1021.85 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 1019.25 | 1021.58 | 1021.61 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 10:15:00 | 1034.05 | 1024.08 | 1022.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 12:15:00 | 1042.85 | 1030.23 | 1025.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 1022.05 | 1030.59 | 1026.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 14:15:00 | 1022.05 | 1030.59 | 1026.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 1022.05 | 1030.59 | 1026.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:00:00 | 1022.05 | 1030.59 | 1026.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 1032.00 | 1030.87 | 1027.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 993.50 | 1030.87 | 1027.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 1011.50 | 1027.00 | 1025.97 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 10:15:00 | 1012.00 | 1024.00 | 1024.70 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 1019.25 | 1013.22 | 1012.66 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 1009.25 | 1012.59 | 1012.76 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 1019.25 | 1013.51 | 1013.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 12:15:00 | 1029.30 | 1018.28 | 1015.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 10:15:00 | 1015.65 | 1018.84 | 1016.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 10:15:00 | 1015.65 | 1018.84 | 1016.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1015.65 | 1018.84 | 1016.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 1015.65 | 1018.84 | 1016.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1020.10 | 1019.09 | 1017.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 09:30:00 | 1030.45 | 1019.87 | 1018.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 11:00:00 | 1023.65 | 1020.63 | 1018.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 12:15:00 | 1023.70 | 1020.63 | 1018.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 14:15:00 | 1024.95 | 1020.23 | 1019.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 1028.50 | 1021.88 | 1019.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 1074.90 | 1023.80 | 1020.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 10:15:00 | 1020.70 | 1045.09 | 1046.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 1020.70 | 1045.09 | 1046.63 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 1038.80 | 1033.18 | 1032.80 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 1028.15 | 1033.02 | 1033.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1006.60 | 1019.91 | 1025.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 15:15:00 | 1009.05 | 1008.75 | 1012.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-11 09:15:00 | 1012.30 | 1008.75 | 1012.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1007.00 | 1008.40 | 1012.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 10:30:00 | 1003.70 | 1006.70 | 1011.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 12:30:00 | 1003.80 | 1005.51 | 1009.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 14:00:00 | 1003.45 | 1005.10 | 1009.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-13 09:15:00 | 903.33 | 969.74 | 986.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 09:15:00 | 972.00 | 952.69 | 951.19 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 14:15:00 | 944.90 | 956.43 | 956.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 10:15:00 | 943.60 | 952.19 | 954.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 942.30 | 941.52 | 946.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 13:00:00 | 942.30 | 941.52 | 946.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 948.95 | 940.97 | 944.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:00:00 | 940.85 | 941.88 | 944.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 14:45:00 | 940.75 | 941.15 | 943.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 13:45:00 | 942.65 | 940.22 | 941.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-26 10:15:00 | 946.35 | 942.66 | 942.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 10:15:00 | 946.35 | 942.66 | 942.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 11:15:00 | 957.00 | 945.53 | 943.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 951.00 | 951.72 | 947.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 15:00:00 | 951.00 | 951.72 | 947.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 946.25 | 950.63 | 947.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 954.90 | 950.63 | 947.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 10:30:00 | 951.95 | 949.67 | 947.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 12:00:00 | 955.65 | 950.86 | 948.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 917.60 | 944.58 | 946.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 917.60 | 944.58 | 946.04 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 966.20 | 943.09 | 940.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 975.40 | 963.82 | 953.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 982.00 | 982.95 | 977.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 982.00 | 982.95 | 977.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 982.00 | 982.95 | 977.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 975.00 | 982.95 | 977.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 972.60 | 980.20 | 977.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:00:00 | 972.60 | 980.20 | 977.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 977.25 | 979.61 | 977.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 13:15:00 | 981.20 | 979.61 | 977.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 11:15:00 | 982.10 | 979.11 | 977.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:00:00 | 978.10 | 978.91 | 977.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 13:15:00 | 973.10 | 977.21 | 977.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 13:15:00 | 973.10 | 977.21 | 977.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 14:15:00 | 965.50 | 974.87 | 976.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 962.20 | 961.79 | 965.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 962.20 | 961.79 | 965.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 962.20 | 961.79 | 965.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 14:30:00 | 954.90 | 957.61 | 961.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 907.15 | 951.37 | 958.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 938.15 | 931.38 | 937.46 | SL hit (close>ema200) qty=0.50 sl=931.38 alert=retest2 |

### Cycle 57 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 934.50 | 922.67 | 921.12 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 910.85 | 927.01 | 928.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 14:15:00 | 904.20 | 911.33 | 917.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 15:15:00 | 876.00 | 873.12 | 883.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 09:15:00 | 872.05 | 873.12 | 883.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 861.50 | 870.80 | 881.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:15:00 | 856.80 | 870.80 | 881.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 12:45:00 | 857.00 | 864.62 | 875.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 914.25 | 876.50 | 876.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 914.25 | 876.50 | 876.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 955.40 | 925.31 | 916.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 930.00 | 930.68 | 921.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 930.00 | 930.68 | 921.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 931.00 | 932.83 | 928.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 929.90 | 932.83 | 928.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 935.00 | 933.26 | 928.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 928.55 | 933.26 | 928.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 964.05 | 974.92 | 969.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 964.05 | 974.92 | 969.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 964.70 | 972.87 | 969.16 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 959.00 | 966.07 | 966.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 951.00 | 963.06 | 965.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 10:15:00 | 959.95 | 956.35 | 959.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 10:15:00 | 959.95 | 956.35 | 959.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 959.95 | 956.35 | 959.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 959.95 | 956.35 | 959.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 958.45 | 956.77 | 959.10 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 976.05 | 960.70 | 960.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 1037.25 | 984.81 | 972.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1047.30 | 1087.00 | 1049.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1047.30 | 1087.00 | 1049.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1047.30 | 1087.00 | 1049.99 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 949.00 | 1015.94 | 1022.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 888.00 | 955.00 | 989.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 985.70 | 946.50 | 965.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 985.70 | 946.50 | 965.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 985.70 | 946.50 | 965.30 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 13:15:00 | 979.95 | 971.02 | 969.94 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 10:15:00 | 959.55 | 972.01 | 973.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 12:15:00 | 954.25 | 959.31 | 964.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 952.20 | 949.72 | 954.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 952.20 | 949.72 | 954.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 952.20 | 949.72 | 954.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 954.45 | 949.72 | 954.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 953.90 | 950.55 | 954.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 12:30:00 | 948.65 | 949.96 | 953.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 09:45:00 | 949.90 | 947.92 | 951.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 14:15:00 | 901.22 | 906.73 | 914.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 14:15:00 | 902.40 | 906.73 | 914.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 904.00 | 903.69 | 910.25 | SL hit (close>ema200) qty=0.50 sl=903.69 alert=retest2 |

### Cycle 65 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 925.00 | 901.59 | 899.06 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 12:15:00 | 901.55 | 902.68 | 902.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 897.55 | 901.65 | 902.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 900.45 | 889.87 | 892.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 900.45 | 889.87 | 892.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 900.45 | 889.87 | 892.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:30:00 | 906.30 | 889.87 | 892.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 902.75 | 892.45 | 893.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 903.35 | 892.45 | 893.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 900.80 | 894.12 | 893.88 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 888.75 | 894.08 | 894.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 886.25 | 892.51 | 893.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 11:15:00 | 894.00 | 890.30 | 891.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 894.00 | 890.30 | 891.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 894.00 | 890.30 | 891.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 894.00 | 890.30 | 891.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 895.25 | 891.29 | 891.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:00:00 | 895.25 | 891.29 | 891.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 892.50 | 891.53 | 891.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:15:00 | 891.00 | 891.53 | 891.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 14:15:00 | 895.05 | 892.24 | 892.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 895.05 | 892.24 | 892.21 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 891.45 | 892.45 | 892.48 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 14:15:00 | 893.70 | 892.70 | 892.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 898.30 | 894.51 | 893.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 12:15:00 | 892.90 | 894.19 | 893.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 12:15:00 | 892.90 | 894.19 | 893.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 892.90 | 894.19 | 893.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:00:00 | 892.90 | 894.19 | 893.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 892.30 | 893.81 | 893.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:30:00 | 895.15 | 893.81 | 893.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 894.45 | 893.94 | 893.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 911.50 | 893.95 | 893.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:00:00 | 899.00 | 901.87 | 899.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 899.15 | 900.98 | 898.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 14:15:00 | 889.65 | 896.72 | 897.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 889.65 | 896.72 | 897.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 888.60 | 895.10 | 896.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 894.00 | 893.34 | 895.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 12:00:00 | 894.00 | 893.34 | 895.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 890.50 | 890.33 | 892.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 890.50 | 890.33 | 892.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 895.35 | 888.23 | 889.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 897.70 | 888.23 | 889.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 900.50 | 890.68 | 890.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 899.25 | 890.68 | 890.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 883.30 | 883.86 | 886.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 886.00 | 883.86 | 886.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 892.85 | 885.66 | 887.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 892.85 | 885.66 | 887.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 890.80 | 886.69 | 887.73 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 892.75 | 888.79 | 888.56 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 884.70 | 888.70 | 888.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 13:15:00 | 879.95 | 886.20 | 887.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 887.00 | 884.98 | 886.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 887.00 | 884.98 | 886.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 887.00 | 884.98 | 886.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 886.75 | 884.98 | 886.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 885.50 | 885.08 | 886.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 886.95 | 885.08 | 886.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 884.25 | 884.92 | 886.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:45:00 | 886.10 | 884.92 | 886.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 897.30 | 887.39 | 887.25 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 871.70 | 897.93 | 901.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 867.00 | 887.56 | 895.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 879.30 | 874.61 | 884.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 12:00:00 | 879.30 | 874.61 | 884.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 888.00 | 877.29 | 884.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:00:00 | 888.00 | 877.29 | 884.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 876.05 | 877.04 | 883.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:15:00 | 874.35 | 877.04 | 883.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 874.70 | 873.69 | 879.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:00:00 | 872.50 | 874.73 | 878.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 874.65 | 878.33 | 878.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 880.10 | 878.47 | 878.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 877.00 | 878.47 | 878.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-12 09:15:00 | 786.92 | 862.80 | 870.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 854.50 | 847.80 | 847.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 861.10 | 855.43 | 852.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 857.00 | 857.05 | 854.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 09:15:00 | 859.35 | 857.05 | 854.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 858.05 | 858.70 | 856.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 858.05 | 858.70 | 856.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 861.50 | 859.52 | 857.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:45:00 | 856.05 | 859.52 | 857.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 858.00 | 859.22 | 857.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:15:00 | 862.90 | 859.24 | 857.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 12:15:00 | 862.30 | 860.19 | 858.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:00:00 | 862.45 | 860.64 | 858.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:45:00 | 862.50 | 861.30 | 859.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 864.90 | 861.99 | 859.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 856.70 | 859.20 | 859.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 856.70 | 859.20 | 859.48 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 13:15:00 | 861.50 | 859.90 | 859.73 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 854.60 | 858.84 | 859.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 853.10 | 856.57 | 857.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 856.85 | 856.37 | 857.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 15:00:00 | 856.85 | 856.37 | 857.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 855.00 | 856.10 | 857.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 853.45 | 855.99 | 857.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 854.30 | 856.19 | 857.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 852.00 | 855.40 | 856.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 850.60 | 843.15 | 842.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 850.60 | 843.15 | 842.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 12:15:00 | 854.00 | 850.12 | 847.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 849.85 | 851.30 | 849.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 849.85 | 851.30 | 849.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 849.85 | 851.30 | 849.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 849.85 | 851.30 | 849.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 848.80 | 850.65 | 849.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:00:00 | 848.80 | 850.65 | 849.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 848.40 | 850.20 | 849.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 848.40 | 850.20 | 849.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 846.00 | 849.36 | 848.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 846.00 | 849.36 | 848.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 15:15:00 | 843.00 | 848.09 | 848.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 834.35 | 845.34 | 847.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 818.65 | 818.23 | 826.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 14:30:00 | 815.90 | 817.42 | 823.05 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:30:00 | 815.50 | 816.05 | 820.92 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 805.50 | 804.80 | 809.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 806.15 | 804.80 | 809.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 816.80 | 805.56 | 807.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 816.80 | 805.56 | 807.09 | SL hit (close>ema400) qty=1.00 sl=807.09 alert=retest1 |

### Cycle 83 — BUY (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 11:15:00 | 814.55 | 809.04 | 808.51 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 801.80 | 807.96 | 808.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 10:15:00 | 799.90 | 806.35 | 807.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 784.40 | 782.08 | 788.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 784.40 | 782.08 | 788.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 787.80 | 783.23 | 788.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 787.25 | 783.23 | 788.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 792.05 | 784.99 | 788.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:00:00 | 792.05 | 784.99 | 788.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 791.00 | 786.19 | 788.88 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 840.95 | 798.07 | 793.70 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 802.70 | 809.99 | 810.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 14:15:00 | 800.20 | 805.94 | 808.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 801.00 | 796.22 | 800.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 801.00 | 796.22 | 800.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 801.00 | 796.22 | 800.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 801.00 | 796.22 | 800.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 796.90 | 796.35 | 800.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:45:00 | 795.00 | 796.13 | 799.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 755.25 | 765.49 | 773.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 750.30 | 748.04 | 757.74 | SL hit (close>ema200) qty=0.50 sl=748.04 alert=retest2 |

### Cycle 87 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 771.10 | 760.80 | 760.25 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 758.90 | 760.41 | 760.45 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 770.35 | 761.67 | 760.79 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 757.65 | 760.31 | 760.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 755.80 | 758.52 | 759.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 749.65 | 748.63 | 752.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 749.65 | 748.63 | 752.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 749.65 | 748.63 | 752.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 753.40 | 748.63 | 752.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 741.70 | 746.31 | 749.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:00:00 | 738.35 | 744.72 | 748.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:45:00 | 738.85 | 743.50 | 747.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 701.91 | 715.89 | 724.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 701.43 | 712.61 | 722.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 14:15:00 | 699.90 | 699.29 | 707.65 | SL hit (close>ema200) qty=0.50 sl=699.29 alert=retest2 |

### Cycle 91 — BUY (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 14:15:00 | 764.70 | 718.49 | 713.57 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 712.70 | 717.22 | 717.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 708.10 | 713.85 | 715.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 711.75 | 711.72 | 713.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 09:15:00 | 714.85 | 711.72 | 713.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 722.65 | 713.91 | 714.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 726.20 | 713.91 | 714.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 723.00 | 715.73 | 715.32 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 710.95 | 717.06 | 717.27 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 723.00 | 716.81 | 716.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 729.90 | 719.43 | 717.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 736.20 | 736.49 | 729.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:45:00 | 737.35 | 736.49 | 729.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 732.20 | 734.89 | 730.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 732.20 | 734.89 | 730.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 726.55 | 733.22 | 730.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 726.55 | 733.22 | 730.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 723.60 | 731.30 | 729.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 721.75 | 731.30 | 729.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 719.40 | 727.43 | 728.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 716.90 | 725.32 | 727.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 708.50 | 707.67 | 713.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 13:30:00 | 698.30 | 705.27 | 710.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 692.90 | 700.97 | 707.24 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 663.38 | 682.15 | 689.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 658.25 | 682.15 | 689.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 682.25 | 674.52 | 680.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 682.25 | 674.52 | 680.70 | SL hit (close>ema200) qty=0.50 sl=674.52 alert=retest1 |

### Cycle 97 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 674.55 | 617.06 | 611.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 14:15:00 | 695.30 | 642.30 | 624.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 774.40 | 799.27 | 763.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 10:00:00 | 774.40 | 799.27 | 763.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 770.25 | 778.35 | 769.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:45:00 | 763.35 | 778.35 | 769.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 774.25 | 777.53 | 770.06 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 10:15:00 | 751.65 | 765.37 | 766.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 12:15:00 | 743.80 | 758.09 | 763.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 13:15:00 | 719.50 | 716.83 | 722.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 13:15:00 | 719.50 | 716.83 | 722.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 719.50 | 716.83 | 722.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:30:00 | 722.55 | 716.83 | 722.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 717.80 | 717.03 | 722.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 716.25 | 717.03 | 722.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 713.40 | 716.71 | 721.23 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 11:15:00 | 730.65 | 725.24 | 724.68 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 716.45 | 724.95 | 725.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 707.85 | 714.32 | 716.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 674.15 | 673.46 | 680.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 13:00:00 | 674.15 | 673.46 | 680.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 678.35 | 673.56 | 677.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 678.35 | 673.56 | 677.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 670.75 | 673.00 | 677.14 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 679.00 | 677.17 | 677.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 695.65 | 680.86 | 678.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 15:15:00 | 746.00 | 749.59 | 735.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:15:00 | 745.90 | 749.59 | 735.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 736.95 | 745.75 | 736.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 736.95 | 745.75 | 736.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 734.40 | 743.48 | 736.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 735.70 | 743.48 | 736.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 734.20 | 741.63 | 735.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:00:00 | 734.20 | 741.63 | 735.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 734.90 | 740.28 | 735.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:30:00 | 736.55 | 740.28 | 735.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 736.35 | 739.49 | 735.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 14:30:00 | 735.50 | 739.49 | 735.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 735.00 | 738.60 | 735.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 742.40 | 738.60 | 735.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 12:30:00 | 737.90 | 738.57 | 736.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 14:15:00 | 727.80 | 735.48 | 735.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 727.80 | 735.48 | 735.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 725.80 | 733.54 | 734.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 704.00 | 701.85 | 708.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 704.00 | 701.85 | 708.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 706.95 | 702.87 | 707.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 706.95 | 702.87 | 707.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 679.00 | 689.21 | 697.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 665.45 | 682.47 | 689.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 632.18 | 650.27 | 668.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 668.40 | 651.58 | 666.04 | SL hit (close>ema200) qty=0.50 sl=651.58 alert=retest2 |

### Cycle 103 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 676.50 | 670.71 | 670.50 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 670.00 | 673.26 | 673.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 650.90 | 666.35 | 669.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 655.90 | 655.17 | 662.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 14:45:00 | 656.95 | 655.17 | 662.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 658.55 | 655.86 | 661.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 658.55 | 655.86 | 661.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 658.55 | 656.40 | 660.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 658.55 | 656.40 | 660.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 661.90 | 658.09 | 660.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 664.35 | 658.09 | 660.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 657.25 | 657.92 | 660.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 656.00 | 657.92 | 660.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 651.70 | 655.26 | 658.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 623.20 | 633.58 | 642.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 619.12 | 633.58 | 642.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 624.20 | 621.64 | 629.23 | SL hit (close>ema200) qty=0.50 sl=621.64 alert=retest2 |

### Cycle 105 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 636.35 | 628.93 | 628.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 642.60 | 631.66 | 629.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 625.95 | 630.52 | 629.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 625.95 | 630.52 | 629.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 625.95 | 630.52 | 629.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 625.95 | 630.52 | 629.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 643.90 | 633.19 | 630.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 633.00 | 633.19 | 630.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 642.65 | 643.66 | 639.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 640.80 | 643.66 | 639.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 629.20 | 640.76 | 638.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 629.20 | 640.76 | 638.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 636.80 | 639.97 | 638.44 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 626.90 | 635.68 | 636.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 622.60 | 631.72 | 634.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 627.55 | 626.59 | 630.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 627.55 | 626.59 | 630.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 627.55 | 626.59 | 630.59 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 637.55 | 630.86 | 630.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 15:15:00 | 642.00 | 637.09 | 634.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 633.25 | 637.03 | 635.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 13:15:00 | 633.25 | 637.03 | 635.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 633.25 | 637.03 | 635.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 633.40 | 637.03 | 635.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 633.95 | 636.41 | 635.17 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 632.15 | 634.47 | 634.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 623.65 | 630.71 | 632.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 600.50 | 598.13 | 606.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 600.50 | 598.13 | 606.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 603.70 | 598.94 | 603.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 604.15 | 598.94 | 603.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 603.35 | 599.82 | 603.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 603.35 | 599.82 | 603.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 602.65 | 600.39 | 603.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 605.90 | 600.39 | 603.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 602.10 | 600.73 | 603.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 601.15 | 600.73 | 603.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 571.09 | 584.65 | 593.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 574.20 | 573.10 | 581.00 | SL hit (close>ema200) qty=0.50 sl=573.10 alert=retest2 |

### Cycle 109 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 583.25 | 579.35 | 578.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 15:15:00 | 583.75 | 580.23 | 579.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 576.70 | 579.52 | 579.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 576.70 | 579.52 | 579.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 576.70 | 579.52 | 579.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 15:15:00 | 588.35 | 583.59 | 581.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:30:00 | 591.05 | 584.71 | 582.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 589.60 | 585.13 | 582.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 575.30 | 582.14 | 582.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 575.30 | 582.14 | 582.36 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 13:15:00 | 584.95 | 579.73 | 579.70 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 15:15:00 | 577.50 | 579.33 | 579.52 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 589.60 | 581.57 | 580.52 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 569.30 | 579.87 | 580.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 562.70 | 575.05 | 577.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 551.70 | 550.15 | 559.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 551.70 | 550.15 | 559.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 555.20 | 549.73 | 553.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 558.65 | 549.73 | 553.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 572.40 | 554.27 | 555.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 572.40 | 554.27 | 555.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 575.05 | 558.42 | 557.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 582.65 | 563.27 | 559.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 606.10 | 607.46 | 599.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 15:15:00 | 604.70 | 607.46 | 599.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 597.75 | 605.08 | 599.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:45:00 | 603.75 | 605.40 | 600.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 590.70 | 597.59 | 598.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 590.70 | 597.59 | 598.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 585.30 | 595.13 | 597.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 601.00 | 594.22 | 596.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 601.00 | 594.22 | 596.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 601.00 | 594.22 | 596.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 601.00 | 594.22 | 596.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 598.95 | 595.17 | 596.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 599.55 | 595.17 | 596.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 612.95 | 598.72 | 597.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 619.05 | 609.56 | 606.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 623.20 | 623.78 | 618.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:00:00 | 623.20 | 623.78 | 618.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 617.95 | 622.36 | 618.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 623.05 | 622.36 | 618.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 616.35 | 624.79 | 625.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 616.35 | 624.79 | 625.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 608.15 | 618.27 | 621.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 617.15 | 616.09 | 619.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 09:30:00 | 608.25 | 616.09 | 619.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 604.60 | 603.90 | 609.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 604.60 | 603.90 | 609.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 608.90 | 604.90 | 609.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 608.90 | 604.90 | 609.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 609.95 | 605.91 | 609.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 614.35 | 605.91 | 609.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 612.05 | 607.14 | 609.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 622.25 | 607.14 | 609.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 627.90 | 611.29 | 611.21 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 606.05 | 611.31 | 611.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 601.00 | 608.19 | 609.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 593.40 | 592.20 | 597.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:00:00 | 593.40 | 592.20 | 597.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 597.00 | 593.84 | 597.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 597.00 | 593.84 | 597.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 599.10 | 594.89 | 597.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:30:00 | 600.55 | 594.89 | 597.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 597.00 | 595.31 | 597.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 598.75 | 595.31 | 597.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 595.05 | 595.26 | 597.47 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 608.55 | 600.04 | 599.21 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 592.40 | 597.86 | 598.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 561.75 | 587.70 | 593.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 574.05 | 572.99 | 581.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 587.20 | 572.99 | 581.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 580.65 | 574.52 | 581.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 577.15 | 576.60 | 582.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 578.00 | 581.37 | 582.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 592.70 | 582.83 | 582.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 592.70 | 582.83 | 582.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 601.70 | 592.23 | 588.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 605.35 | 605.72 | 599.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 14:00:00 | 609.65 | 606.83 | 601.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:15:00 | 614.10 | 606.77 | 602.72 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 607.60 | 609.58 | 606.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 610.25 | 609.58 | 606.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 622.90 | 624.69 | 622.99 | SL hit (close<ema400) qty=1.00 sl=622.99 alert=retest1 |

### Cycle 124 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 609.40 | 621.49 | 621.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 599.65 | 617.12 | 619.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 611.25 | 607.48 | 612.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 611.25 | 607.48 | 612.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 611.25 | 607.48 | 612.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 611.25 | 607.48 | 612.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 618.50 | 609.68 | 613.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 618.50 | 609.68 | 613.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 618.50 | 611.44 | 613.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 619.50 | 611.44 | 613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 616.20 | 614.01 | 614.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:30:00 | 615.55 | 614.01 | 614.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 618.00 | 614.81 | 614.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 619.75 | 614.81 | 614.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 613.65 | 614.44 | 614.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 613.65 | 614.44 | 614.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 610.75 | 604.90 | 607.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 610.75 | 604.90 | 607.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 606.35 | 605.19 | 607.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 603.85 | 605.19 | 607.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 629.85 | 607.79 | 607.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 629.85 | 607.79 | 607.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 655.75 | 617.38 | 611.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 646.00 | 649.71 | 635.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:00:00 | 646.00 | 649.71 | 635.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 637.15 | 645.47 | 636.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 637.15 | 645.47 | 636.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 634.30 | 643.24 | 636.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 631.60 | 643.24 | 636.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 631.90 | 640.97 | 636.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 633.00 | 640.97 | 636.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 627.30 | 635.38 | 634.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 629.85 | 633.87 | 633.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 12:15:00 | 623.80 | 631.85 | 632.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 623.80 | 631.85 | 632.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 617.00 | 627.41 | 629.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 604.55 | 602.71 | 612.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 626.15 | 602.71 | 612.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 628.55 | 607.88 | 613.54 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 628.80 | 618.35 | 617.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 631.75 | 622.86 | 619.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 652.00 | 652.25 | 645.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 652.00 | 652.25 | 645.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 672.55 | 675.64 | 670.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 671.00 | 675.64 | 670.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 669.55 | 674.43 | 670.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 669.55 | 674.43 | 670.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 669.00 | 673.34 | 670.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 669.00 | 673.34 | 670.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 667.10 | 672.09 | 670.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 666.20 | 672.09 | 670.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 659.85 | 667.98 | 668.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 655.50 | 660.58 | 663.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 660.25 | 660.01 | 662.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 660.25 | 660.01 | 662.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 660.25 | 660.01 | 662.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 665.90 | 660.01 | 662.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 661.80 | 660.37 | 662.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 661.55 | 660.37 | 662.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 662.65 | 660.83 | 662.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 662.65 | 660.83 | 662.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 664.30 | 661.52 | 662.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 663.75 | 661.52 | 662.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 658.55 | 661.00 | 662.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 662.10 | 661.00 | 662.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 668.05 | 661.82 | 662.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 670.05 | 661.82 | 662.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 673.85 | 664.23 | 663.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 679.00 | 671.01 | 667.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 675.50 | 676.68 | 673.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:45:00 | 676.00 | 676.68 | 673.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 672.00 | 675.33 | 673.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 669.35 | 675.33 | 673.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 672.00 | 674.66 | 673.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 671.65 | 674.66 | 673.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 673.20 | 674.37 | 673.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 671.55 | 674.37 | 673.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 671.50 | 673.80 | 673.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:45:00 | 672.30 | 673.80 | 673.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 681.35 | 675.31 | 673.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 671.30 | 675.31 | 673.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 679.40 | 676.09 | 674.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 684.00 | 676.09 | 674.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 682.35 | 677.29 | 675.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 682.00 | 678.03 | 676.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 681.75 | 678.03 | 676.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 685.35 | 683.35 | 680.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:30:00 | 686.70 | 684.42 | 680.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:00:00 | 688.70 | 684.42 | 680.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 691.10 | 687.01 | 683.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 678.50 | 682.04 | 682.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 678.50 | 682.04 | 682.18 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 688.55 | 682.91 | 682.31 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 679.75 | 682.05 | 682.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 11:15:00 | 677.35 | 681.11 | 681.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 15:15:00 | 681.50 | 680.83 | 681.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 15:15:00 | 681.50 | 680.83 | 681.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 681.50 | 680.83 | 681.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 685.80 | 680.83 | 681.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 685.90 | 681.84 | 681.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 685.90 | 681.84 | 681.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 686.80 | 682.83 | 682.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 697.30 | 689.67 | 686.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 707.10 | 707.39 | 699.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:00:00 | 707.10 | 707.39 | 699.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 694.95 | 704.19 | 699.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 695.65 | 704.19 | 699.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 698.00 | 702.95 | 699.74 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 685.05 | 696.77 | 697.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 683.95 | 694.21 | 696.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 660.25 | 659.67 | 668.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 660.25 | 659.67 | 668.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 659.70 | 652.74 | 658.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 659.70 | 652.74 | 658.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 659.00 | 653.99 | 658.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 652.60 | 654.87 | 658.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 619.97 | 625.18 | 634.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 624.35 | 624.35 | 632.53 | SL hit (close>ema200) qty=0.50 sl=624.35 alert=retest2 |

### Cycle 135 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 636.55 | 632.37 | 632.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 644.00 | 639.09 | 636.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 672.75 | 673.30 | 660.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 672.75 | 673.30 | 660.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 670.95 | 674.52 | 669.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 670.95 | 674.52 | 669.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 666.55 | 672.93 | 668.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 666.55 | 672.93 | 668.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 666.80 | 671.70 | 668.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 665.75 | 671.70 | 668.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 664.50 | 667.94 | 667.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 662.60 | 667.94 | 667.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 665.00 | 667.58 | 667.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 663.70 | 666.80 | 667.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 667.50 | 666.12 | 666.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 667.50 | 666.12 | 666.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 667.50 | 666.12 | 666.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 667.50 | 666.12 | 666.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 665.50 | 665.99 | 666.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 664.00 | 665.99 | 666.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 663.85 | 665.56 | 666.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 663.65 | 665.56 | 665.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 654.85 | 649.02 | 648.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 654.85 | 649.02 | 648.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 659.55 | 652.71 | 650.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 657.45 | 659.81 | 656.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 657.45 | 659.81 | 656.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 656.40 | 659.12 | 656.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 654.25 | 659.12 | 656.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 655.65 | 658.43 | 656.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 655.65 | 658.43 | 656.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 655.95 | 657.93 | 656.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 657.75 | 656.52 | 656.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 657.65 | 657.40 | 656.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 657.65 | 657.68 | 656.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:30:00 | 659.00 | 657.68 | 656.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 658.55 | 657.86 | 657.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 654.00 | 657.08 | 656.82 | SL hit (close<static) qty=1.00 sl=654.80 alert=retest2 |

### Cycle 138 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 651.30 | 655.93 | 656.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 648.00 | 652.62 | 654.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 651.65 | 651.53 | 653.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 651.65 | 651.53 | 653.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 651.65 | 651.53 | 653.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 646.95 | 649.83 | 651.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 646.95 | 649.33 | 651.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 646.00 | 649.33 | 651.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:15:00 | 614.60 | 623.12 | 626.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:15:00 | 614.60 | 623.12 | 626.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 613.70 | 617.53 | 623.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 604.50 | 603.31 | 608.72 | SL hit (close>ema200) qty=0.50 sl=603.31 alert=retest2 |

### Cycle 139 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 596.70 | 592.45 | 591.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 605.80 | 595.69 | 593.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 607.70 | 607.96 | 602.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 607.70 | 607.96 | 602.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 607.35 | 609.04 | 607.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 607.60 | 609.04 | 607.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 608.70 | 608.97 | 607.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 611.70 | 608.97 | 607.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 618.00 | 624.34 | 624.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 618.00 | 624.34 | 624.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 616.25 | 620.26 | 622.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 624.90 | 620.14 | 621.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 624.90 | 620.14 | 621.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 624.90 | 620.14 | 621.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 624.05 | 620.14 | 621.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 625.60 | 621.23 | 621.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 627.10 | 621.23 | 621.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 626.90 | 622.37 | 622.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 12:15:00 | 631.30 | 624.15 | 623.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 633.00 | 635.00 | 631.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 617.05 | 635.00 | 631.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 609.25 | 629.85 | 629.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 608.20 | 629.85 | 629.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 619.95 | 627.87 | 628.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 593.20 | 614.75 | 621.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 596.30 | 595.30 | 603.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:45:00 | 596.25 | 595.30 | 603.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 598.50 | 596.11 | 599.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 597.40 | 597.23 | 599.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 597.80 | 596.95 | 598.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 597.80 | 597.04 | 598.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 597.65 | 597.21 | 598.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 598.25 | 597.42 | 598.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 598.00 | 597.42 | 598.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 596.35 | 597.21 | 598.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 595.00 | 597.17 | 597.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 599.95 | 594.10 | 595.11 | SL hit (close>static) qty=1.00 sl=598.45 alert=retest2 |

### Cycle 143 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 600.85 | 595.45 | 595.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 609.40 | 599.99 | 597.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 606.00 | 606.72 | 603.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:00:00 | 606.00 | 606.72 | 603.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 605.40 | 605.78 | 604.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 611.65 | 605.90 | 605.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 609.80 | 607.69 | 606.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 667.40 | 607.14 | 606.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 09:15:00 | 672.82 | 617.98 | 611.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 678.20 | 702.18 | 705.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 666.65 | 695.08 | 701.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 630.15 | 628.34 | 640.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 637.85 | 628.34 | 640.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 642.15 | 632.16 | 637.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 642.15 | 632.16 | 637.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 641.20 | 633.96 | 637.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 644.05 | 633.96 | 637.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 636.25 | 635.94 | 637.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 635.90 | 635.94 | 637.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 634.65 | 635.69 | 637.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 633.90 | 635.69 | 637.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:30:00 | 633.60 | 635.35 | 636.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 634.00 | 634.08 | 635.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 633.80 | 634.08 | 635.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 641.40 | 635.50 | 635.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 641.40 | 635.50 | 635.94 | SL hit (close>static) qty=1.00 sl=639.05 alert=retest2 |

### Cycle 145 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 11:15:00 | 637.20 | 636.41 | 636.31 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 629.95 | 635.16 | 635.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 629.20 | 633.97 | 635.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 629.85 | 629.20 | 630.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 629.85 | 629.20 | 630.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 629.85 | 629.20 | 630.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 631.65 | 629.20 | 630.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 628.20 | 629.00 | 630.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 623.30 | 628.67 | 629.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 626.65 | 626.05 | 627.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 625.55 | 622.29 | 621.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 625.55 | 622.29 | 621.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 636.95 | 625.22 | 623.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 624.90 | 625.15 | 623.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 624.90 | 625.15 | 623.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 622.00 | 624.52 | 623.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 623.50 | 624.52 | 623.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 622.95 | 624.21 | 623.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 622.50 | 624.21 | 623.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 620.05 | 623.38 | 622.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 620.05 | 623.38 | 622.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 619.55 | 622.61 | 622.66 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 625.05 | 622.22 | 622.13 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 620.05 | 622.61 | 622.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 619.00 | 620.64 | 621.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 620.60 | 620.06 | 620.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 620.60 | 620.06 | 620.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 620.60 | 620.06 | 620.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 622.20 | 620.06 | 620.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 618.95 | 619.84 | 620.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 620.20 | 619.84 | 620.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 621.40 | 620.15 | 620.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 621.40 | 620.15 | 620.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 620.75 | 620.27 | 620.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 619.35 | 620.18 | 620.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 619.55 | 620.18 | 620.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 616.50 | 620.74 | 620.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 626.40 | 621.87 | 621.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 626.40 | 621.87 | 621.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 658.10 | 629.12 | 624.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 634.30 | 637.95 | 631.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 634.30 | 637.95 | 631.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 638.00 | 637.12 | 631.88 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 631.20 | 633.48 | 633.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 626.15 | 629.79 | 631.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 629.50 | 628.78 | 630.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 629.50 | 628.78 | 630.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 629.50 | 628.78 | 630.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 628.45 | 628.78 | 630.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 630.15 | 628.28 | 629.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 630.15 | 628.28 | 629.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 628.15 | 628.26 | 629.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 621.50 | 628.26 | 629.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 629.55 | 615.38 | 614.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 629.55 | 615.38 | 614.65 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 618.30 | 622.00 | 622.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 613.45 | 620.29 | 621.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 615.80 | 612.58 | 614.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 615.80 | 612.58 | 614.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 615.80 | 612.58 | 614.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 608.05 | 613.63 | 614.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 606.20 | 605.66 | 605.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 606.20 | 605.66 | 605.60 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 604.05 | 605.38 | 605.48 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 611.25 | 606.34 | 605.71 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 601.70 | 605.69 | 606.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 601.30 | 604.23 | 605.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 594.60 | 593.49 | 595.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 594.60 | 593.49 | 595.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 594.60 | 593.49 | 595.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 594.45 | 593.49 | 595.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 582.30 | 591.15 | 593.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 579.85 | 587.15 | 591.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 579.25 | 587.15 | 591.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 594.60 | 587.06 | 586.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 594.60 | 587.06 | 586.62 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 585.75 | 586.98 | 587.01 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 587.80 | 587.15 | 587.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 591.50 | 588.20 | 587.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 592.10 | 592.44 | 591.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 592.10 | 592.44 | 591.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 587.95 | 591.63 | 590.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 588.50 | 591.63 | 590.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 588.80 | 591.07 | 590.76 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 587.05 | 590.26 | 590.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 586.00 | 588.13 | 589.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 582.25 | 580.89 | 583.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 582.25 | 580.89 | 583.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 582.45 | 581.21 | 583.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 583.60 | 581.21 | 583.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 578.00 | 580.56 | 582.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 576.00 | 578.86 | 581.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:00:00 | 572.05 | 577.30 | 580.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 576.85 | 569.61 | 569.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 576.85 | 569.61 | 569.14 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 566.60 | 569.37 | 569.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 566.05 | 568.71 | 569.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 565.50 | 564.96 | 566.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 565.50 | 564.96 | 566.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 563.90 | 564.75 | 566.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 565.75 | 564.75 | 566.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 566.45 | 565.13 | 566.14 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 567.60 | 566.72 | 566.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 613.60 | 576.15 | 571.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 591.00 | 591.59 | 582.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 09:15:00 | 589.05 | 591.59 | 582.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 578.80 | 586.91 | 584.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 578.00 | 586.91 | 584.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 578.80 | 585.29 | 584.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 578.80 | 585.29 | 584.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 579.15 | 582.95 | 583.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 577.20 | 581.80 | 582.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 571.30 | 571.15 | 574.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 571.30 | 571.15 | 574.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 572.95 | 571.66 | 573.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 574.65 | 571.66 | 573.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 555.75 | 554.51 | 558.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 557.40 | 554.51 | 558.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 559.00 | 555.41 | 558.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 558.40 | 555.41 | 558.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 557.55 | 555.84 | 558.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 554.85 | 556.13 | 558.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 554.60 | 555.85 | 557.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 554.35 | 555.47 | 557.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 554.70 | 555.54 | 556.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 554.70 | 555.37 | 556.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 555.15 | 555.37 | 556.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 549.90 | 554.12 | 555.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 547.00 | 550.28 | 553.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 527.11 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 526.87 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 526.63 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 526.97 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 539.05 | 532.27 | 536.14 | SL hit (close>ema200) qty=0.50 sl=532.27 alert=retest2 |

### Cycle 167 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 546.15 | 538.91 | 538.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 551.00 | 541.33 | 539.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 538.00 | 540.66 | 539.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 538.00 | 540.66 | 539.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 538.00 | 540.66 | 539.30 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 524.00 | 536.09 | 537.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 517.95 | 532.46 | 535.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 535.35 | 528.89 | 532.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 535.35 | 528.89 | 532.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 535.35 | 528.89 | 532.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 535.35 | 528.89 | 532.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 526.00 | 528.31 | 532.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:00:00 | 524.70 | 527.96 | 531.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 524.35 | 526.22 | 529.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 524.25 | 525.02 | 528.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 524.40 | 527.13 | 528.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 532.45 | 526.94 | 527.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:45:00 | 530.80 | 526.94 | 527.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 528.30 | 527.21 | 527.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:15:00 | 525.05 | 527.21 | 527.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 527.00 | 527.26 | 527.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 527.00 | 527.04 | 527.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 15:15:00 | 529.90 | 527.59 | 527.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 529.90 | 527.59 | 527.52 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 525.50 | 527.32 | 527.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 522.95 | 526.45 | 527.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 518.35 | 517.31 | 520.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 518.35 | 517.31 | 520.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 171 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 546.35 | 523.53 | 522.99 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 543.00 | 545.16 | 545.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 540.20 | 544.17 | 544.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 542.25 | 542.20 | 543.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 542.25 | 542.20 | 543.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 536.50 | 541.07 | 542.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 535.00 | 538.34 | 540.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 533.20 | 529.91 | 529.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 533.20 | 529.91 | 529.80 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 528.50 | 530.52 | 530.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 526.80 | 529.78 | 530.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 525.40 | 522.69 | 524.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 525.40 | 522.69 | 524.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 525.40 | 522.69 | 524.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 527.50 | 522.69 | 524.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 525.00 | 523.15 | 524.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 525.60 | 523.15 | 524.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 523.60 | 523.24 | 524.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 520.35 | 524.15 | 524.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 494.33 | 509.73 | 513.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 482.10 | 482.02 | 487.79 | SL hit (close>ema200) qty=0.50 sl=482.02 alert=retest2 |

### Cycle 175 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 537.45 | 485.36 | 480.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 551.15 | 507.28 | 491.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 13:15:00 | 576.45 | 601.60 | 579.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 13:15:00 | 576.45 | 601.60 | 579.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 576.45 | 601.60 | 579.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 576.45 | 601.60 | 579.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 574.00 | 596.08 | 579.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 574.00 | 596.08 | 579.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 572.50 | 591.36 | 578.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 550.30 | 591.36 | 578.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 532.50 | 572.62 | 571.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:00:00 | 532.50 | 572.62 | 571.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 527.00 | 563.50 | 567.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 519.10 | 554.62 | 563.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 09:15:00 | 553.50 | 523.99 | 527.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 553.50 | 523.99 | 527.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 553.50 | 523.99 | 527.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 568.30 | 523.99 | 527.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 10:15:00 | 564.55 | 532.10 | 531.10 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 520.40 | 542.23 | 543.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 515.50 | 533.37 | 539.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 519.45 | 519.13 | 527.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 518.55 | 519.13 | 527.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 526.05 | 521.63 | 526.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 15:00:00 | 521.50 | 523.24 | 525.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 538.80 | 526.13 | 526.34 | SL hit (close>static) qty=1.00 sl=531.40 alert=retest2 |

### Cycle 179 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 531.45 | 527.20 | 526.80 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 515.10 | 525.43 | 526.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 513.10 | 520.44 | 523.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 517.05 | 516.08 | 520.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 517.05 | 516.08 | 520.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 517.05 | 516.08 | 520.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 512.05 | 519.86 | 520.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 10:15:00 | 513.10 | 518.85 | 520.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 521.60 | 519.96 | 519.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 521.60 | 519.96 | 519.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 523.40 | 520.65 | 520.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 521.30 | 522.22 | 521.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 521.30 | 522.22 | 521.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 521.30 | 522.22 | 521.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 521.30 | 522.22 | 521.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 521.20 | 522.01 | 521.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 521.20 | 522.01 | 521.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 520.75 | 521.76 | 521.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 520.40 | 521.76 | 521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 518.90 | 521.19 | 521.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 518.30 | 521.19 | 521.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 519.00 | 520.75 | 520.85 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 531.00 | 522.84 | 521.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 566.15 | 531.50 | 525.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 548.00 | 549.10 | 539.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 558.40 | 550.20 | 544.06 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 554.70 | 551.10 | 545.03 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:15:00 | 582.44 | 557.39 | 548.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 580.50 | 572.63 | 560.68 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:15:00 | 586.32 | 575.53 | 563.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 587.10 | 575.53 | 563.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 12:15:00 | 577.50 | 577.70 | 566.34 | SL hit (close<ema200) qty=0.50 sl=577.70 alert=retest1 |

### Cycle 184 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 612.35 | 621.60 | 621.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 610.10 | 619.30 | 620.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 626.00 | 618.09 | 619.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 626.00 | 618.09 | 619.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 626.00 | 618.09 | 619.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 627.50 | 618.09 | 619.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 625.00 | 619.48 | 620.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 627.70 | 619.48 | 620.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 633.15 | 622.21 | 621.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 643.00 | 626.37 | 623.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 657.30 | 657.42 | 649.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 646.65 | 657.42 | 649.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 629.15 | 651.77 | 647.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 629.15 | 651.77 | 647.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 628.95 | 647.20 | 646.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 628.65 | 647.20 | 646.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 632.20 | 644.20 | 644.81 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 648.00 | 643.48 | 642.96 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 638.80 | 642.35 | 642.78 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 648.00 | 643.73 | 643.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 650.20 | 645.78 | 644.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 646.90 | 647.14 | 645.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 11:15:00 | 646.90 | 647.14 | 645.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 646.90 | 647.14 | 645.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 645.60 | 647.14 | 645.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 646.35 | 646.94 | 645.82 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 641.60 | 645.18 | 645.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 635.65 | 641.10 | 643.16 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-06-13 09:15:00 | 667.00 | 2023-06-15 12:15:00 | 669.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-06-15 11:30:00 | 666.65 | 2023-06-15 12:15:00 | 669.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-06-22 11:30:00 | 657.60 | 2023-06-26 09:15:00 | 624.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-22 11:30:00 | 657.60 | 2023-06-26 09:15:00 | 640.00 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2023-07-07 09:15:00 | 645.00 | 2023-07-17 09:15:00 | 647.55 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-07-10 14:30:00 | 643.80 | 2023-07-17 09:15:00 | 647.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-07-27 15:00:00 | 655.70 | 2023-08-02 13:15:00 | 649.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-07-28 09:15:00 | 656.00 | 2023-08-02 13:15:00 | 649.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-08-08 11:15:00 | 655.00 | 2023-08-08 13:15:00 | 655.80 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2023-08-08 12:30:00 | 653.70 | 2023-08-08 13:15:00 | 655.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-08-10 10:30:00 | 646.40 | 2023-08-18 09:15:00 | 641.20 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2023-08-10 14:30:00 | 645.55 | 2023-08-18 09:15:00 | 641.20 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2023-08-10 15:00:00 | 646.95 | 2023-08-18 09:15:00 | 641.20 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2023-08-14 09:15:00 | 636.00 | 2023-08-18 11:15:00 | 662.55 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2023-08-17 10:45:00 | 633.85 | 2023-08-18 11:15:00 | 662.55 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2023-08-17 12:15:00 | 633.60 | 2023-08-18 11:15:00 | 662.55 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2023-08-17 14:30:00 | 634.25 | 2023-08-18 11:15:00 | 662.55 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2023-08-25 10:15:00 | 650.95 | 2023-08-29 13:15:00 | 659.75 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-08-25 10:45:00 | 652.00 | 2023-08-29 13:15:00 | 659.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-08-25 14:15:00 | 652.00 | 2023-08-29 13:15:00 | 659.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-08-28 10:30:00 | 650.85 | 2023-08-29 13:15:00 | 659.75 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2023-09-04 14:15:00 | 633.15 | 2023-09-06 14:15:00 | 642.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-09-05 09:30:00 | 633.75 | 2023-09-06 14:15:00 | 642.95 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2023-09-05 10:00:00 | 633.95 | 2023-09-06 14:15:00 | 642.95 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-09-05 12:30:00 | 632.75 | 2023-09-06 14:15:00 | 642.95 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-09-11 09:45:00 | 648.95 | 2023-09-12 14:15:00 | 642.85 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-09-12 10:30:00 | 648.85 | 2023-09-12 14:15:00 | 642.85 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-09-12 13:15:00 | 647.00 | 2023-09-12 14:15:00 | 642.85 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-09-15 15:00:00 | 635.70 | 2023-09-21 10:15:00 | 644.85 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-09-18 14:15:00 | 640.00 | 2023-09-21 10:15:00 | 644.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-09-18 14:45:00 | 639.95 | 2023-09-21 10:15:00 | 644.85 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-09-27 11:00:00 | 628.95 | 2023-10-09 09:15:00 | 597.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-27 11:45:00 | 629.00 | 2023-10-09 09:15:00 | 597.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-27 12:15:00 | 629.00 | 2023-10-09 09:15:00 | 597.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-27 11:00:00 | 628.95 | 2023-10-10 09:15:00 | 594.20 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2023-09-27 11:45:00 | 629.00 | 2023-10-10 09:15:00 | 594.20 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2023-09-27 12:15:00 | 629.00 | 2023-10-10 09:15:00 | 594.20 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2023-10-18 11:00:00 | 594.55 | 2023-10-25 12:15:00 | 564.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:45:00 | 594.20 | 2023-10-25 12:15:00 | 564.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 12:30:00 | 594.50 | 2023-10-25 12:15:00 | 564.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:45:00 | 594.40 | 2023-10-25 12:15:00 | 564.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:45:00 | 590.90 | 2023-10-25 12:15:00 | 561.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 13:15:00 | 591.20 | 2023-10-25 12:15:00 | 561.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 13:45:00 | 591.05 | 2023-10-25 12:15:00 | 561.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 15:15:00 | 589.40 | 2023-10-25 12:15:00 | 559.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 10:15:00 | 578.50 | 2023-10-25 14:15:00 | 549.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:00:00 | 594.55 | 2023-10-26 09:15:00 | 535.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 11:45:00 | 594.20 | 2023-10-26 09:15:00 | 534.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 12:30:00 | 594.50 | 2023-10-26 09:15:00 | 535.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 09:45:00 | 594.40 | 2023-10-26 09:15:00 | 534.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 12:45:00 | 590.90 | 2023-10-26 09:15:00 | 531.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 13:15:00 | 591.20 | 2023-10-26 09:15:00 | 532.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 13:45:00 | 591.05 | 2023-10-26 09:15:00 | 531.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 15:15:00 | 589.40 | 2023-10-26 09:15:00 | 530.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-23 10:15:00 | 578.50 | 2023-10-27 09:15:00 | 555.10 | STOP_HIT | 0.50 | 4.04% |
| BUY | retest1 | 2023-10-31 09:15:00 | 570.75 | 2023-11-01 10:15:00 | 557.85 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-11-01 09:15:00 | 567.00 | 2023-11-01 10:15:00 | 557.85 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-11-03 14:30:00 | 555.50 | 2023-11-13 14:15:00 | 542.30 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2023-11-06 09:30:00 | 553.85 | 2023-11-13 14:15:00 | 542.30 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2023-11-20 11:30:00 | 528.35 | 2023-11-21 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2023-11-20 12:30:00 | 528.95 | 2023-11-21 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2023-11-20 13:00:00 | 528.40 | 2023-11-21 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2023-12-05 10:15:00 | 769.10 | 2023-12-05 12:15:00 | 846.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-18 09:30:00 | 1033.90 | 2023-12-20 13:15:00 | 930.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-18 10:30:00 | 1029.65 | 2023-12-20 13:15:00 | 978.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-18 10:30:00 | 1029.65 | 2023-12-20 13:15:00 | 926.77 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2023-12-18 15:00:00 | 1029.75 | 2023-12-20 13:15:00 | 931.05 | TARGET_HIT | 1.00 | 9.58% |
| SELL | retest2 | 2023-12-19 13:30:00 | 1034.50 | 2023-12-20 13:15:00 | 927.54 | TARGET_HIT | 1.00 | 10.34% |
| SELL | retest2 | 2023-12-20 11:15:00 | 1030.60 | 2023-12-21 09:15:00 | 926.69 | TARGET_HIT | 1.00 | 10.08% |
| SELL | retest2 | 2023-12-29 13:15:00 | 986.80 | 2024-01-01 10:15:00 | 1000.10 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-12-29 14:45:00 | 986.00 | 2024-01-01 10:15:00 | 1000.10 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-12-29 15:15:00 | 986.80 | 2024-01-01 10:15:00 | 1000.10 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-01-01 09:30:00 | 987.00 | 2024-01-01 10:15:00 | 1000.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-01-12 12:30:00 | 1060.50 | 2024-01-17 09:15:00 | 1007.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-12 13:15:00 | 1060.00 | 2024-01-17 09:15:00 | 1007.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-12 14:30:00 | 1061.00 | 2024-01-17 09:15:00 | 1007.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-12 12:30:00 | 1060.50 | 2024-01-19 09:15:00 | 999.75 | STOP_HIT | 0.50 | 5.73% |
| SELL | retest2 | 2024-01-12 13:15:00 | 1060.00 | 2024-01-19 09:15:00 | 999.75 | STOP_HIT | 0.50 | 5.68% |
| SELL | retest2 | 2024-01-12 14:30:00 | 1061.00 | 2024-01-19 09:15:00 | 999.75 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2024-02-02 14:15:00 | 1002.10 | 2024-02-07 09:15:00 | 1049.60 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2024-02-02 15:00:00 | 999.65 | 2024-02-07 09:15:00 | 1049.60 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2024-02-23 09:30:00 | 1030.45 | 2024-02-28 10:15:00 | 1020.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-02-23 11:00:00 | 1023.65 | 2024-02-28 10:15:00 | 1020.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-02-23 12:15:00 | 1023.70 | 2024-02-28 10:15:00 | 1020.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-02-23 14:15:00 | 1024.95 | 2024-02-28 10:15:00 | 1020.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-02-26 09:15:00 | 1074.90 | 2024-02-28 10:15:00 | 1020.70 | STOP_HIT | 1.00 | -5.04% |
| SELL | retest2 | 2024-03-11 10:30:00 | 1003.70 | 2024-03-13 09:15:00 | 903.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-11 12:30:00 | 1003.80 | 2024-03-13 09:15:00 | 903.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-11 14:00:00 | 1003.45 | 2024-03-13 09:15:00 | 903.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-21 12:00:00 | 940.85 | 2024-03-26 10:15:00 | 946.35 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-03-21 14:45:00 | 940.75 | 2024-03-26 10:15:00 | 946.35 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-03-22 13:45:00 | 942.65 | 2024-03-26 10:15:00 | 946.35 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-03-27 09:15:00 | 954.90 | 2024-03-27 14:15:00 | 917.60 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2024-03-27 10:30:00 | 951.95 | 2024-03-27 14:15:00 | 917.60 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2024-03-27 12:00:00 | 955.65 | 2024-03-27 14:15:00 | 917.60 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2024-04-05 13:15:00 | 981.20 | 2024-04-08 13:15:00 | 973.10 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-04-08 11:15:00 | 982.10 | 2024-04-08 13:15:00 | 973.10 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-04-08 12:00:00 | 978.10 | 2024-04-08 13:15:00 | 973.10 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-04-12 14:30:00 | 954.90 | 2024-04-15 09:15:00 | 907.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 14:30:00 | 954.90 | 2024-04-18 09:15:00 | 938.15 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2024-05-13 10:15:00 | 856.80 | 2024-05-14 11:15:00 | 914.25 | STOP_HIT | 1.00 | -6.71% |
| SELL | retest2 | 2024-05-13 12:45:00 | 857.00 | 2024-05-14 11:15:00 | 914.25 | STOP_HIT | 1.00 | -6.68% |
| SELL | retest2 | 2024-06-14 12:30:00 | 948.65 | 2024-06-25 14:15:00 | 901.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-18 09:45:00 | 949.90 | 2024-06-25 14:15:00 | 902.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-14 12:30:00 | 948.65 | 2024-06-26 11:15:00 | 904.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2024-06-18 09:45:00 | 949.90 | 2024-06-26 11:15:00 | 904.00 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-07-11 14:15:00 | 891.00 | 2024-07-11 14:15:00 | 895.05 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-07-16 09:15:00 | 911.50 | 2024-07-18 14:15:00 | 889.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-07-18 10:00:00 | 899.00 | 2024-07-18 14:15:00 | 889.65 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-18 11:15:00 | 899.15 | 2024-07-18 14:15:00 | 889.65 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-08-06 14:15:00 | 874.35 | 2024-08-12 09:15:00 | 786.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-07 11:00:00 | 874.70 | 2024-08-12 09:15:00 | 787.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-07 14:00:00 | 872.50 | 2024-08-12 09:15:00 | 785.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-08 15:00:00 | 874.65 | 2024-08-12 09:15:00 | 787.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-09 10:15:00 | 877.00 | 2024-08-12 09:15:00 | 789.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-22 10:15:00 | 862.90 | 2024-08-26 09:15:00 | 856.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-22 12:15:00 | 862.30 | 2024-08-26 09:15:00 | 856.70 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-08-22 13:00:00 | 862.45 | 2024-08-26 09:15:00 | 856.70 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-22 13:45:00 | 862.50 | 2024-08-26 09:15:00 | 856.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-08-28 09:30:00 | 853.45 | 2024-09-03 09:15:00 | 850.60 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-08-28 10:30:00 | 854.30 | 2024-09-03 09:15:00 | 850.60 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2024-08-28 12:30:00 | 852.00 | 2024-09-03 09:15:00 | 850.60 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest1 | 2024-09-10 14:30:00 | 815.90 | 2024-09-16 09:15:00 | 816.80 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2024-09-11 10:30:00 | 815.50 | 2024-09-16 09:15:00 | 816.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-09-27 11:45:00 | 795.00 | 2024-10-07 09:15:00 | 755.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 11:45:00 | 795.00 | 2024-10-08 10:15:00 | 750.30 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2024-10-17 11:00:00 | 738.35 | 2024-10-22 11:15:00 | 701.91 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-10-17 11:45:00 | 738.85 | 2024-10-22 12:15:00 | 701.43 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2024-10-17 11:00:00 | 738.35 | 2024-10-23 14:15:00 | 699.90 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2024-10-17 11:45:00 | 738.85 | 2024-10-23 14:15:00 | 699.90 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest1 | 2024-11-12 13:30:00 | 698.30 | 2024-11-18 09:15:00 | 663.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 692.90 | 2024-11-18 09:15:00 | 658.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-12 13:30:00 | 698.30 | 2024-11-19 09:15:00 | 682.25 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest1 | 2024-11-13 09:30:00 | 692.90 | 2024-11-19 09:15:00 | 682.25 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2024-11-19 14:00:00 | 676.90 | 2024-11-21 09:15:00 | 609.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-03 09:15:00 | 742.40 | 2025-01-03 14:15:00 | 727.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-01-03 12:30:00 | 737.90 | 2025-01-03 14:15:00 | 727.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-13 09:15:00 | 665.45 | 2025-01-13 14:15:00 | 632.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 09:15:00 | 665.45 | 2025-01-14 09:15:00 | 668.40 | STOP_HIT | 0.50 | -0.44% |
| SELL | retest2 | 2025-01-14 15:15:00 | 668.60 | 2025-01-16 10:15:00 | 676.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-15 10:15:00 | 667.95 | 2025-01-16 10:15:00 | 676.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-01-15 10:45:00 | 668.25 | 2025-01-16 10:15:00 | 676.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-01-23 15:15:00 | 656.00 | 2025-01-27 14:15:00 | 623.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 651.70 | 2025-01-27 14:15:00 | 619.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 656.00 | 2025-01-29 09:15:00 | 624.20 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2025-01-24 09:30:00 | 651.70 | 2025-01-29 09:15:00 | 624.20 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-02-13 13:15:00 | 601.15 | 2025-02-14 13:15:00 | 571.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 601.15 | 2025-02-17 14:15:00 | 574.20 | STOP_HIT | 0.50 | 4.48% |
| BUY | retest2 | 2025-02-20 15:15:00 | 588.35 | 2025-02-24 09:15:00 | 575.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-02-21 10:30:00 | 591.05 | 2025-02-24 09:15:00 | 575.30 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-02-21 11:30:00 | 589.60 | 2025-02-24 09:15:00 | 575.30 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-03-11 10:45:00 | 603.75 | 2025-03-12 10:15:00 | 590.70 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-03-21 09:15:00 | 623.05 | 2025-03-25 10:15:00 | 616.35 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-04-08 10:30:00 | 577.15 | 2025-04-11 09:15:00 | 592.70 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-04-09 11:00:00 | 578.00 | 2025-04-11 09:15:00 | 592.70 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2025-04-16 14:00:00 | 609.65 | 2025-04-24 13:15:00 | 622.90 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest1 | 2025-04-17 10:15:00 | 614.10 | 2025-04-24 13:15:00 | 622.90 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-04-21 09:15:00 | 610.25 | 2025-04-25 09:15:00 | 609.40 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-05-02 11:15:00 | 603.85 | 2025-05-05 09:15:00 | 629.85 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2025-05-07 11:30:00 | 629.85 | 2025-05-07 12:15:00 | 623.80 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-05-29 15:15:00 | 684.00 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-05-30 11:15:00 | 682.35 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-30 11:45:00 | 682.00 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-05-30 12:15:00 | 681.75 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-06-02 12:30:00 | 686.70 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-02 13:00:00 | 688.70 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-03 09:45:00 | 691.10 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-18 12:15:00 | 652.60 | 2025-06-20 14:15:00 | 619.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 652.60 | 2025-06-23 09:15:00 | 624.35 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-07-03 11:15:00 | 664.00 | 2025-07-16 09:15:00 | 654.85 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-07-03 12:00:00 | 663.85 | 2025-07-16 09:15:00 | 654.85 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2025-07-04 11:30:00 | 663.65 | 2025-07-16 09:15:00 | 654.85 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-07-21 10:00:00 | 657.75 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-21 11:45:00 | 657.65 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-21 14:00:00 | 657.65 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-21 14:30:00 | 659.00 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-24 12:45:00 | 646.95 | 2025-07-31 11:15:00 | 614.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:30:00 | 646.95 | 2025-07-31 11:15:00 | 614.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:15:00 | 646.00 | 2025-07-31 14:15:00 | 613.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 12:45:00 | 646.95 | 2025-08-04 13:15:00 | 604.50 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-07-24 13:30:00 | 646.95 | 2025-08-04 13:15:00 | 604.50 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-07-24 14:15:00 | 646.00 | 2025-08-04 13:15:00 | 604.50 | STOP_HIT | 0.50 | 6.42% |
| BUY | retest2 | 2025-08-18 09:15:00 | 611.70 | 2025-08-21 14:15:00 | 618.00 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-09-02 13:15:00 | 597.40 | 2025-09-08 09:15:00 | 599.95 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-09-03 09:30:00 | 597.80 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-03 10:30:00 | 597.80 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-03 11:45:00 | 597.65 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-04 15:15:00 | 595.00 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-08 13:30:00 | 595.20 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-08 14:45:00 | 593.50 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-09 09:30:00 | 594.80 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-09 13:45:00 | 592.30 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-09-09 14:15:00 | 593.00 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-09-17 09:15:00 | 611.65 | 2025-09-19 09:15:00 | 672.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-18 11:00:00 | 609.80 | 2025-09-19 09:15:00 | 670.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-19 09:15:00 | 667.40 | 2025-09-22 10:15:00 | 734.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-03 12:15:00 | 633.90 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-06 10:30:00 | 633.60 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-06 14:30:00 | 634.00 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-06 15:15:00 | 633.80 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-13 09:15:00 | 623.30 | 2025-10-17 09:15:00 | 625.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-14 09:45:00 | 626.65 | 2025-10-17 09:15:00 | 625.55 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-10-28 13:30:00 | 619.35 | 2025-10-29 09:15:00 | 626.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-28 14:15:00 | 619.55 | 2025-10-29 09:15:00 | 626.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-29 09:15:00 | 616.50 | 2025-10-29 09:15:00 | 626.40 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-11-06 09:15:00 | 621.50 | 2025-11-12 09:15:00 | 629.55 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-11-20 15:15:00 | 608.05 | 2025-11-26 14:15:00 | 606.20 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-12-08 11:30:00 | 579.85 | 2025-12-10 09:15:00 | 594.60 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-12-08 12:00:00 | 579.25 | 2025-12-10 09:15:00 | 594.60 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-19 12:30:00 | 576.00 | 2025-12-26 11:15:00 | 576.85 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-19 15:00:00 | 572.05 | 2025-12-26 11:15:00 | 576.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-01-13 12:00:00 | 554.85 | 2026-01-21 09:15:00 | 527.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 554.60 | 2026-01-21 09:15:00 | 526.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 554.35 | 2026-01-21 09:15:00 | 526.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:45:00 | 554.70 | 2026-01-21 09:15:00 | 526.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 554.85 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-01-13 14:15:00 | 554.60 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-01-14 10:00:00 | 554.35 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2026-01-14 13:45:00 | 554.70 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-01-16 15:00:00 | 547.00 | 2026-01-22 14:15:00 | 546.15 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2026-01-27 13:00:00 | 524.70 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-28 09:15:00 | 524.35 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-28 12:15:00 | 524.25 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-29 10:15:00 | 524.40 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-29 15:15:00 | 525.05 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-30 12:15:00 | 527.00 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-30 14:15:00 | 527.00 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-02-12 15:15:00 | 535.00 | 2026-02-17 12:15:00 | 533.20 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2026-02-24 09:15:00 | 520.35 | 2026-03-02 09:15:00 | 494.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 520.35 | 2026-03-05 14:15:00 | 482.10 | STOP_HIT | 0.50 | 7.35% |
| SELL | retest2 | 2026-03-25 15:00:00 | 521.50 | 2026-03-27 09:15:00 | 538.80 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2026-04-02 09:15:00 | 512.05 | 2026-04-06 11:15:00 | 521.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-04-02 10:15:00 | 513.10 | 2026-04-06 11:15:00 | 521.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2026-04-10 09:15:00 | 558.40 | 2026-04-10 11:15:00 | 582.44 | PARTIAL | 0.50 | 4.30% |
| BUY | retest1 | 2026-04-10 10:00:00 | 554.70 | 2026-04-13 10:15:00 | 586.32 | PARTIAL | 0.50 | 5.70% |
| BUY | retest1 | 2026-04-10 09:15:00 | 558.40 | 2026-04-13 12:15:00 | 577.50 | STOP_HIT | 0.50 | 3.42% |
| BUY | retest1 | 2026-04-10 10:00:00 | 554.70 | 2026-04-13 12:15:00 | 577.50 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest2 | 2026-04-13 11:00:00 | 587.10 | 2026-04-17 14:15:00 | 645.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 10:00:00 | 587.15 | 2026-04-17 14:15:00 | 645.87 | TARGET_HIT | 1.00 | 10.00% |
