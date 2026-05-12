# Swan Corp Ltd. (SWANCORP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 353.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 12 |
| ALERT2_SKIP | 5 |
| ALERT3 | 64 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 50 |
| PARTIAL | 10 |
| TARGET_HIT | 1 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 7
- **Winners / losers:** 19 / 34
- **Target hits / Stop hits / Partials:** 1 / 42 / 10
- **Avg / median % per leg:** -0.07% / -1.44%
- **Sum % (uncompounded):** -3.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.90% | -26.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.90% | -26.1% |
| SELL (all) | 44 | 19 | 43.2% | 1 | 33 | 10 | 0.51% | 22.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 19 | 43.2% | 1 | 33 | 10 | 0.51% | 22.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 19 | 35.8% | 1 | 42 | 10 | -0.07% | -3.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 651.00 | 609.09 | 609.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 14:15:00 | 655.00 | 611.99 | 610.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 614.55 | 615.02 | 612.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 614.55 | 615.02 | 612.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 614.55 | 615.02 | 612.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 612.00 | 615.02 | 612.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 603.00 | 614.91 | 612.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 603.00 | 614.91 | 612.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 587.40 | 614.64 | 612.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:30:00 | 590.00 | 614.64 | 612.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 613.00 | 611.73 | 610.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:30:00 | 609.00 | 611.73 | 610.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 666.00 | 691.90 | 665.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 14:00:00 | 666.00 | 691.90 | 665.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 653.45 | 691.52 | 665.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 653.45 | 691.52 | 665.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 662.70 | 691.23 | 665.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 670.10 | 691.23 | 665.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 666.00 | 686.28 | 674.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 665.00 | 686.06 | 674.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 648.05 | 683.32 | 673.81 | SL hit (close<static) qty=1.00 sl=651.65 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 620.45 | 665.83 | 665.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 607.45 | 656.32 | 660.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 10:15:00 | 543.20 | 529.29 | 566.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 11:00:00 | 543.20 | 529.29 | 566.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 561.15 | 529.61 | 566.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 557.50 | 529.61 | 566.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 562.00 | 529.93 | 566.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:45:00 | 566.35 | 529.93 | 566.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 557.40 | 530.14 | 559.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 557.40 | 530.14 | 559.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 569.25 | 530.88 | 557.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 569.25 | 530.88 | 557.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 584.40 | 531.41 | 557.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 584.40 | 531.41 | 557.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 714.50 | 576.90 | 576.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 731.30 | 587.49 | 581.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 703.90 | 704.60 | 664.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 13:45:00 | 704.40 | 704.60 | 664.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 663.50 | 703.42 | 668.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 663.50 | 703.42 | 668.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 670.50 | 703.09 | 668.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 667.40 | 703.09 | 668.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 666.40 | 702.72 | 668.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 666.40 | 702.72 | 668.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 663.90 | 702.34 | 668.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:30:00 | 664.05 | 702.34 | 668.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 655.10 | 701.87 | 668.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 655.10 | 701.87 | 668.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 666.00 | 687.65 | 664.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 672.00 | 687.65 | 664.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 658.30 | 687.36 | 664.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 658.30 | 687.36 | 664.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 673.50 | 687.22 | 664.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:30:00 | 677.00 | 687.05 | 664.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:45:00 | 675.70 | 686.92 | 664.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:15:00 | 680.80 | 686.64 | 664.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 646.50 | 686.31 | 665.20 | SL hit (close<static) qty=1.00 sl=656.95 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 15:15:00 | 529.30 | 648.98 | 649.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 517.55 | 600.45 | 621.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 453.15 | 451.22 | 501.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-24 10:00:00 | 453.15 | 451.22 | 501.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 438.30 | 422.87 | 446.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:30:00 | 443.55 | 422.87 | 446.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 441.50 | 423.94 | 446.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:45:00 | 443.75 | 423.94 | 446.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 444.60 | 424.15 | 446.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:45:00 | 446.25 | 424.15 | 446.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 444.95 | 424.36 | 446.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:30:00 | 446.00 | 424.36 | 446.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 456.10 | 424.67 | 446.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 456.10 | 424.67 | 446.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 456.85 | 424.99 | 446.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 09:30:00 | 450.65 | 425.59 | 446.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 453.50 | 426.45 | 446.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 14:15:00 | 430.82 | 427.77 | 446.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 430.75 | 427.77 | 446.16 | SL hit (close>static) qty=0.50 sl=427.77 alert=retest2 |

### Cycle 5 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 451.95 | 444.05 | 444.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 464.15 | 444.46 | 444.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 472.40 | 472.80 | 460.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 472.40 | 472.80 | 460.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 471.40 | 473.40 | 462.25 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 429.70 | 454.33 | 454.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 426.60 | 454.06 | 454.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 463.05 | 448.30 | 451.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 13:15:00 | 463.05 | 448.30 | 451.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 463.05 | 448.30 | 451.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:45:00 | 462.05 | 448.30 | 451.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 455.00 | 448.36 | 451.13 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 469.20 | 453.67 | 453.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 475.00 | 454.95 | 454.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 09:15:00 | 463.60 | 467.14 | 461.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 463.60 | 467.14 | 461.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 464.20 | 467.11 | 461.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 462.15 | 467.11 | 461.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 457.70 | 467.01 | 461.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 459.25 | 467.01 | 461.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 455.75 | 466.90 | 461.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 455.75 | 466.90 | 461.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 453.00 | 466.53 | 461.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 454.75 | 466.53 | 461.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 454.30 | 465.85 | 461.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:45:00 | 454.25 | 464.47 | 461.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 448.85 | 464.09 | 460.89 | SL hit (close<static) qty=1.00 sl=449.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 433.50 | 457.98 | 458.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 432.00 | 457.03 | 457.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 449.85 | 446.43 | 451.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 449.85 | 446.43 | 451.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 449.85 | 446.43 | 451.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:30:00 | 452.10 | 446.43 | 451.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 451.20 | 446.47 | 451.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 459.15 | 446.47 | 451.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 456.20 | 446.57 | 451.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 456.05 | 446.57 | 451.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 459.25 | 446.70 | 451.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 454.80 | 446.93 | 451.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 466.00 | 447.12 | 451.59 | SL hit (close>static) qty=1.00 sl=464.60 alert=retest2 |

### Cycle 9 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 474.95 | 455.42 | 455.37 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 446.30 | 456.33 | 456.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 442.05 | 455.66 | 456.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 454.40 | 454.14 | 455.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 14:15:00 | 454.40 | 454.14 | 455.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 454.40 | 454.14 | 455.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 454.40 | 454.14 | 455.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 458.00 | 452.37 | 454.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 458.00 | 452.37 | 454.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 455.00 | 452.40 | 454.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 452.40 | 452.40 | 454.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 454.90 | 452.49 | 454.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 454.05 | 452.49 | 454.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:00:00 | 454.75 | 452.47 | 454.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 455.80 | 452.50 | 454.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 455.80 | 452.50 | 454.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 459.40 | 452.57 | 454.11 | SL hit (close>static) qty=1.00 sl=459.25 alert=retest2 |

### Cycle 11 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 471.30 | 455.49 | 455.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 476.00 | 456.78 | 456.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 461.50 | 462.27 | 459.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:45:00 | 462.00 | 462.27 | 459.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 459.25 | 463.77 | 460.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 459.25 | 463.77 | 460.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 451.40 | 463.64 | 460.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 451.40 | 463.64 | 460.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 420.10 | 457.62 | 457.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 419.25 | 457.24 | 457.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 364.00 | 338.72 | 364.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 10:15:00 | 364.00 | 338.72 | 364.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 364.00 | 338.72 | 364.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 364.00 | 338.72 | 364.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 359.85 | 338.93 | 364.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 13:30:00 | 355.10 | 339.28 | 364.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 14:15:00 | 356.30 | 339.28 | 364.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:00:00 | 353.10 | 339.42 | 364.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 356.50 | 339.89 | 364.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 337.35 | 342.29 | 361.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 338.49 | 342.29 | 361.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 335.44 | 342.29 | 361.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 338.68 | 342.29 | 361.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 341.85 | 341.25 | 359.87 | SL hit (close>ema200) qty=0.50 sl=341.25 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-22 14:00:00 | 605.55 | 2024-05-24 12:15:00 | 629.00 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-05-23 10:00:00 | 608.00 | 2024-05-24 12:15:00 | 629.00 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-05-23 11:45:00 | 607.00 | 2024-05-24 12:15:00 | 629.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-05-30 09:45:00 | 608.50 | 2024-05-30 12:15:00 | 617.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-05-30 14:45:00 | 610.00 | 2024-05-31 09:15:00 | 579.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 14:45:00 | 610.00 | 2024-06-05 09:15:00 | 549.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-11 12:45:00 | 611.90 | 2024-06-13 12:15:00 | 604.95 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2024-06-11 14:45:00 | 607.80 | 2024-06-13 12:15:00 | 604.95 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2024-06-12 10:15:00 | 610.05 | 2024-06-14 11:15:00 | 620.30 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-06-12 14:00:00 | 601.80 | 2024-06-14 11:15:00 | 620.30 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-06-12 15:15:00 | 598.00 | 2024-06-14 11:15:00 | 620.30 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-08-07 09:15:00 | 670.10 | 2024-09-04 13:15:00 | 648.05 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-09-03 09:15:00 | 666.00 | 2024-09-04 13:15:00 | 648.05 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-09-03 10:00:00 | 665.00 | 2024-09-04 13:15:00 | 648.05 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-01-16 14:30:00 | 677.00 | 2025-01-20 09:15:00 | 646.50 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest2 | 2025-01-17 09:45:00 | 675.70 | 2025-01-20 09:15:00 | 646.50 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2025-01-17 12:15:00 | 680.80 | 2025-01-20 09:15:00 | 646.50 | STOP_HIT | 1.00 | -5.04% |
| SELL | retest2 | 2025-05-19 09:30:00 | 450.65 | 2025-05-20 14:15:00 | 430.82 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-05-19 09:30:00 | 450.65 | 2025-05-20 14:15:00 | 430.75 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-05-19 13:00:00 | 453.50 | 2025-05-21 09:15:00 | 428.12 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-05-19 13:00:00 | 453.50 | 2025-05-21 09:15:00 | 434.95 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-06-04 13:30:00 | 453.60 | 2025-06-05 10:15:00 | 463.05 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-06 09:15:00 | 451.75 | 2025-06-09 13:15:00 | 468.00 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-06-13 09:15:00 | 435.00 | 2025-06-19 12:15:00 | 413.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 10:30:00 | 439.95 | 2025-06-19 12:15:00 | 417.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 09:15:00 | 435.00 | 2025-06-24 09:15:00 | 452.45 | STOP_HIT | 0.50 | -4.01% |
| SELL | retest2 | 2025-06-13 10:30:00 | 439.95 | 2025-06-24 09:15:00 | 452.45 | STOP_HIT | 0.50 | -2.84% |
| SELL | retest2 | 2025-07-01 13:00:00 | 439.70 | 2025-07-03 09:15:00 | 442.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-02 09:30:00 | 439.00 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-07-03 09:15:00 | 437.10 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-09-30 09:15:00 | 454.75 | 2025-10-07 10:15:00 | 448.85 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-30 14:45:00 | 454.30 | 2025-10-07 10:15:00 | 448.85 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-10-06 14:45:00 | 454.25 | 2025-10-07 10:15:00 | 448.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-28 13:15:00 | 454.80 | 2025-10-28 13:15:00 | 466.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-04 15:00:00 | 455.55 | 2025-11-07 09:15:00 | 432.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 15:00:00 | 455.55 | 2025-11-07 10:15:00 | 473.50 | STOP_HIT | 0.50 | -3.94% |
| SELL | retest2 | 2025-11-10 10:45:00 | 454.80 | 2025-11-10 13:15:00 | 470.65 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-12-10 09:15:00 | 452.40 | 2025-12-11 14:15:00 | 459.40 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-12-10 10:45:00 | 454.90 | 2025-12-11 14:15:00 | 459.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-10 11:15:00 | 454.05 | 2025-12-11 14:15:00 | 459.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-11 13:00:00 | 454.75 | 2025-12-11 14:15:00 | 459.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-15 09:30:00 | 453.65 | 2025-12-15 12:15:00 | 464.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-12-15 10:30:00 | 453.35 | 2025-12-15 12:15:00 | 464.55 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-12-15 11:00:00 | 453.65 | 2025-12-15 12:15:00 | 464.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-12-16 11:30:00 | 453.50 | 2025-12-16 13:15:00 | 460.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-16 13:30:00 | 355.10 | 2026-04-24 09:15:00 | 337.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 14:15:00 | 356.30 | 2026-04-24 09:15:00 | 338.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 15:00:00 | 353.10 | 2026-04-24 09:15:00 | 335.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 11:15:00 | 356.50 | 2026-04-24 09:15:00 | 338.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 13:30:00 | 355.10 | 2026-04-28 10:15:00 | 341.85 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2026-04-16 14:15:00 | 356.30 | 2026-04-28 10:15:00 | 341.85 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2026-04-16 15:00:00 | 353.10 | 2026-04-28 10:15:00 | 341.85 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-04-17 11:15:00 | 356.50 | 2026-04-28 10:15:00 | 341.85 | STOP_HIT | 0.50 | 4.11% |
