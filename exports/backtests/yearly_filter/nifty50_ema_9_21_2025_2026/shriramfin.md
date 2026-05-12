# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1003.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 20 |
| ALERT3 | 133 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 40
- **Target hits / Stop hits / Partials:** 1 / 53 / 5
- **Avg / median % per leg:** 0.36% / -0.75%
- **Sum % (uncompounded):** 21.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 9 | 52.9% | 1 | 16 | 0 | 1.34% | 22.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 9 | 52.9% | 1 | 16 | 0 | 1.34% | 22.7% |
| SELL (all) | 42 | 10 | 23.8% | 0 | 37 | 5 | -0.03% | -1.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 10 | 23.8% | 0 | 37 | 5 | -0.03% | -1.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 59 | 19 | 32.2% | 1 | 53 | 5 | 0.36% | 21.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 646.00 | 621.40 | 619.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 654.80 | 639.56 | 633.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 665.45 | 666.50 | 656.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 665.45 | 666.50 | 656.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 673.30 | 672.73 | 667.77 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 659.70 | 665.46 | 665.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 649.65 | 662.30 | 664.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 658.85 | 658.48 | 661.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:30:00 | 655.80 | 658.48 | 661.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 660.05 | 652.57 | 655.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 660.05 | 652.57 | 655.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 658.20 | 653.69 | 655.67 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 659.35 | 657.04 | 656.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 664.90 | 660.35 | 658.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 661.15 | 661.91 | 660.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 13:15:00 | 661.15 | 661.91 | 660.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 661.15 | 661.91 | 660.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 661.15 | 661.91 | 660.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 662.40 | 662.01 | 660.51 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 656.05 | 659.71 | 659.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 655.95 | 658.35 | 659.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 653.60 | 653.29 | 655.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 644.05 | 651.44 | 654.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 644.05 | 651.44 | 654.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 640.50 | 651.44 | 654.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 641.75 | 647.42 | 651.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:45:00 | 642.30 | 646.14 | 650.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 641.95 | 640.84 | 645.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 644.60 | 641.97 | 645.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 644.60 | 641.97 | 645.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 645.75 | 642.80 | 645.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 651.90 | 646.67 | 646.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 651.90 | 646.67 | 646.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 657.20 | 649.26 | 647.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 12:15:00 | 642.60 | 649.59 | 648.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 642.60 | 649.59 | 648.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 642.60 | 649.59 | 648.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 646.05 | 649.59 | 648.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 642.50 | 648.17 | 647.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 642.50 | 648.17 | 647.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 640.95 | 646.73 | 647.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 639.10 | 644.45 | 646.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 648.75 | 644.94 | 645.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 648.75 | 644.94 | 645.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 648.75 | 644.94 | 645.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 648.75 | 644.94 | 645.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 652.65 | 646.48 | 646.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 652.65 | 646.48 | 646.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 651.60 | 647.50 | 647.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 680.80 | 655.76 | 651.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 696.90 | 697.17 | 688.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:45:00 | 696.60 | 697.17 | 688.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 694.95 | 697.61 | 690.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 690.25 | 697.61 | 690.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 687.35 | 694.54 | 691.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 687.35 | 694.54 | 691.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 686.50 | 692.93 | 690.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 686.50 | 692.93 | 690.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 684.60 | 691.26 | 690.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 685.65 | 691.26 | 690.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 677.60 | 687.43 | 688.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 675.55 | 683.43 | 686.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 665.00 | 664.93 | 673.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 665.00 | 664.93 | 673.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 672.85 | 667.28 | 672.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 678.85 | 667.28 | 672.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 677.30 | 669.29 | 672.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 678.95 | 669.29 | 672.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 676.15 | 670.66 | 673.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 673.05 | 673.13 | 673.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 676.30 | 663.81 | 662.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 676.30 | 663.81 | 662.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 685.85 | 677.95 | 674.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 701.70 | 705.08 | 699.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 701.70 | 705.08 | 699.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 700.30 | 704.12 | 699.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 700.30 | 704.12 | 699.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 699.30 | 703.16 | 699.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 699.30 | 703.16 | 699.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 696.45 | 701.82 | 699.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 697.00 | 701.82 | 699.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 697.25 | 700.90 | 699.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 695.90 | 700.90 | 699.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 686.95 | 696.82 | 697.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 682.90 | 694.04 | 696.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 683.80 | 679.09 | 683.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 683.80 | 679.09 | 683.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 683.80 | 679.09 | 683.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 680.50 | 679.09 | 683.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 677.70 | 678.81 | 682.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 675.80 | 678.81 | 682.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 675.90 | 676.40 | 680.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:00:00 | 675.80 | 676.28 | 680.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 675.20 | 676.82 | 680.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 676.65 | 676.59 | 679.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 676.65 | 676.59 | 679.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 672.60 | 673.00 | 676.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 669.65 | 672.25 | 675.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 668.35 | 671.06 | 674.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 679.05 | 672.95 | 673.97 | SL hit (close>static) qty=1.00 sl=677.40 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 682.10 | 675.91 | 675.21 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 673.65 | 676.31 | 676.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 664.95 | 672.76 | 674.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 669.70 | 669.32 | 671.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:15:00 | 668.75 | 669.32 | 671.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 671.00 | 669.36 | 671.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 671.00 | 669.36 | 671.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 668.85 | 669.26 | 670.91 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 679.50 | 672.74 | 671.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 683.15 | 676.86 | 674.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 668.45 | 676.64 | 674.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 668.45 | 676.64 | 674.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 668.45 | 676.64 | 674.75 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 667.75 | 672.93 | 673.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 660.40 | 665.83 | 668.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 651.70 | 650.13 | 656.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 651.70 | 650.13 | 656.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 656.60 | 651.71 | 655.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 656.60 | 651.71 | 655.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 655.70 | 652.51 | 655.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 650.15 | 652.51 | 655.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 647.85 | 651.58 | 655.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 645.80 | 650.42 | 654.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 645.95 | 649.22 | 653.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 636.35 | 648.33 | 648.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 613.51 | 633.60 | 640.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 613.65 | 633.60 | 640.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 604.53 | 623.66 | 634.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 637.25 | 620.30 | 627.79 | SL hit (close>ema200) qty=0.50 sl=620.30 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 639.70 | 631.18 | 630.57 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 626.85 | 632.21 | 632.47 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 636.80 | 633.22 | 632.81 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 623.40 | 631.51 | 632.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 619.40 | 625.51 | 628.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 621.80 | 621.40 | 625.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 621.80 | 621.40 | 625.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 624.65 | 622.44 | 624.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 624.90 | 622.44 | 624.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 625.40 | 623.03 | 624.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 624.10 | 623.03 | 624.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 623.35 | 623.10 | 624.65 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 631.75 | 625.59 | 625.29 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 619.05 | 626.63 | 626.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 616.60 | 623.54 | 624.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 12:15:00 | 616.15 | 614.83 | 618.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 13:00:00 | 616.15 | 614.83 | 618.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 618.05 | 615.48 | 618.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 618.05 | 615.48 | 618.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 617.45 | 615.87 | 617.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 617.45 | 615.87 | 617.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 616.90 | 616.08 | 617.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 614.80 | 615.98 | 617.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 613.85 | 615.98 | 617.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:45:00 | 614.85 | 613.71 | 615.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 619.50 | 616.21 | 616.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 619.50 | 616.21 | 616.01 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 612.85 | 615.73 | 615.89 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 633.75 | 619.16 | 617.30 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 615.80 | 622.12 | 622.19 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 621.75 | 618.85 | 618.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 623.15 | 619.71 | 618.96 | Break + close above crossover candle high |

### Cycle 26 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 609.00 | 617.84 | 618.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 606.30 | 615.54 | 617.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 585.75 | 581.08 | 590.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 584.50 | 581.08 | 590.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 588.85 | 584.32 | 587.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 588.85 | 584.32 | 587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 588.75 | 585.21 | 587.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 588.75 | 585.21 | 587.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 588.90 | 585.95 | 587.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 589.80 | 585.95 | 587.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 592.60 | 587.53 | 587.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 594.10 | 587.53 | 587.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 591.50 | 588.32 | 588.24 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 583.95 | 588.32 | 588.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 581.30 | 585.95 | 587.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 13:15:00 | 585.10 | 584.45 | 586.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 585.10 | 584.45 | 586.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 586.30 | 584.82 | 586.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 586.30 | 584.82 | 586.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 587.25 | 585.31 | 586.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 589.10 | 585.31 | 586.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 587.35 | 585.72 | 586.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 586.90 | 585.95 | 586.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:15:00 | 586.40 | 586.26 | 586.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:00:00 | 586.55 | 586.36 | 586.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 591.30 | 587.35 | 586.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 591.30 | 587.35 | 586.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 592.40 | 589.00 | 587.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 596.10 | 597.94 | 594.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 596.10 | 597.94 | 594.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 591.00 | 596.34 | 594.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 591.00 | 596.34 | 594.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 592.65 | 595.60 | 593.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:45:00 | 594.30 | 594.75 | 593.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 595.65 | 594.75 | 593.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 594.30 | 594.66 | 593.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 615.50 | 620.42 | 620.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 615.50 | 620.42 | 620.76 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 622.70 | 621.14 | 620.96 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 619.70 | 620.80 | 620.84 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 622.70 | 621.18 | 621.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 626.00 | 622.14 | 621.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 11:15:00 | 632.00 | 632.69 | 629.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 632.00 | 632.69 | 629.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 632.40 | 632.63 | 630.04 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 623.90 | 629.06 | 629.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 618.50 | 624.34 | 626.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 611.50 | 610.64 | 615.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 611.50 | 610.64 | 615.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 615.10 | 610.99 | 613.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 615.10 | 610.99 | 613.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 612.30 | 611.25 | 613.78 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 618.70 | 615.31 | 615.10 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 612.90 | 614.87 | 614.94 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 617.85 | 615.55 | 615.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 627.95 | 618.03 | 616.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 664.90 | 665.64 | 656.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:30:00 | 664.85 | 665.64 | 656.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 665.75 | 666.81 | 663.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 672.00 | 668.26 | 666.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 672.95 | 669.56 | 667.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 671.80 | 670.92 | 669.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 664.50 | 672.17 | 672.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 664.50 | 672.17 | 672.31 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 676.05 | 672.56 | 672.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 682.75 | 676.49 | 674.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 712.80 | 713.20 | 704.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:45:00 | 713.75 | 713.20 | 704.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 711.35 | 718.35 | 714.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 711.35 | 718.35 | 714.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 714.85 | 717.65 | 714.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 719.45 | 718.08 | 714.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 09:15:00 | 791.40 | 751.96 | 741.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 814.65 | 818.83 | 819.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 809.50 | 815.56 | 817.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 823.35 | 813.36 | 814.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 823.35 | 813.36 | 814.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 823.35 | 813.36 | 814.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 823.35 | 813.36 | 814.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 825.15 | 815.72 | 815.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 830.35 | 824.17 | 822.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 821.75 | 823.69 | 822.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 821.75 | 823.69 | 822.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 821.75 | 823.69 | 822.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 821.75 | 823.69 | 822.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 822.15 | 823.38 | 822.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 828.90 | 823.38 | 822.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 834.80 | 825.67 | 823.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 838.90 | 825.67 | 823.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 835.20 | 830.47 | 827.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 835.00 | 831.37 | 828.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 851.95 | 854.12 | 854.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 851.95 | 854.12 | 854.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 844.20 | 851.42 | 852.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 828.35 | 828.28 | 833.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 15:15:00 | 831.05 | 828.28 | 833.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 849.40 | 832.95 | 834.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 849.40 | 832.95 | 834.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 848.60 | 836.08 | 836.18 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 850.00 | 838.86 | 837.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 12:15:00 | 853.75 | 841.84 | 838.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 847.60 | 847.96 | 843.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 847.60 | 847.96 | 843.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 841.60 | 846.15 | 843.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 841.60 | 846.15 | 843.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 840.85 | 845.09 | 843.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 840.85 | 845.09 | 843.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 832.35 | 842.54 | 842.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 832.35 | 842.54 | 842.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 834.55 | 840.94 | 841.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 827.75 | 837.09 | 839.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 842.20 | 838.11 | 839.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 842.20 | 838.11 | 839.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 842.20 | 838.11 | 839.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 842.20 | 838.11 | 839.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 841.90 | 838.87 | 839.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 843.00 | 838.87 | 839.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 846.60 | 841.02 | 840.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 850.60 | 844.27 | 842.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 843.35 | 844.34 | 842.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:30:00 | 844.60 | 844.34 | 842.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 841.55 | 843.78 | 842.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 841.35 | 843.78 | 842.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 839.20 | 842.87 | 842.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 839.20 | 842.87 | 842.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 837.30 | 841.75 | 841.91 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 848.80 | 842.40 | 842.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 853.25 | 847.74 | 845.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 855.40 | 855.80 | 851.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 855.40 | 855.80 | 851.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 851.05 | 854.85 | 851.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 850.10 | 854.85 | 851.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 850.70 | 854.02 | 851.42 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 845.80 | 850.39 | 850.57 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 860.40 | 852.39 | 851.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 907.15 | 873.06 | 866.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 959.90 | 966.91 | 951.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 959.90 | 966.91 | 951.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 955.05 | 960.84 | 955.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 955.05 | 960.84 | 955.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 956.40 | 959.95 | 955.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 969.00 | 956.14 | 955.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 993.65 | 1001.69 | 1002.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 993.65 | 1001.69 | 1002.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 988.70 | 995.59 | 998.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 12:15:00 | 994.20 | 993.47 | 996.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 13:00:00 | 994.20 | 993.47 | 996.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 993.00 | 993.38 | 996.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:45:00 | 994.50 | 993.38 | 996.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 993.05 | 993.31 | 996.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 995.80 | 993.31 | 996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 970.05 | 972.27 | 978.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 980.50 | 972.27 | 978.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 978.95 | 973.60 | 978.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:45:00 | 979.75 | 973.60 | 978.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 982.05 | 975.29 | 978.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 984.15 | 975.29 | 978.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 975.35 | 975.30 | 978.52 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 986.30 | 979.71 | 979.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1006.20 | 985.44 | 982.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 995.05 | 997.06 | 990.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 15:00:00 | 995.05 | 997.06 | 990.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1002.00 | 998.03 | 992.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 1012.40 | 1001.31 | 994.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 984.85 | 998.70 | 998.56 | SL hit (close<static) qty=1.00 sl=991.20 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 985.25 | 996.01 | 997.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 982.70 | 993.35 | 996.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 995.05 | 988.94 | 992.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 995.05 | 988.94 | 992.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 995.05 | 988.94 | 992.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1002.15 | 988.94 | 992.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1001.45 | 991.44 | 992.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1002.00 | 991.44 | 992.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 996.80 | 992.51 | 993.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 992.15 | 992.51 | 993.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 1000.60 | 994.97 | 994.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 1000.60 | 994.97 | 994.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1004.70 | 997.61 | 995.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 1004.25 | 1005.00 | 1001.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 1004.25 | 1005.00 | 1001.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1004.25 | 1005.00 | 1001.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:30:00 | 1007.70 | 1005.00 | 1001.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1002.90 | 1004.58 | 1001.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 991.15 | 1004.58 | 1001.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 999.65 | 1003.59 | 1001.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 989.30 | 1003.59 | 1001.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 996.15 | 1002.10 | 1000.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:45:00 | 1000.15 | 1002.10 | 1000.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 995.70 | 1000.82 | 1000.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 995.70 | 1000.82 | 1000.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 991.65 | 998.99 | 999.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 987.70 | 996.73 | 998.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1000.90 | 997.56 | 998.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1000.90 | 997.56 | 998.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1000.90 | 997.56 | 998.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1000.90 | 997.56 | 998.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1000.05 | 998.06 | 998.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1015.00 | 998.06 | 998.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 1004.70 | 999.81 | 999.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 1008.55 | 1001.56 | 1000.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 1019.30 | 1019.52 | 1014.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 1019.30 | 1019.52 | 1014.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1018.55 | 1020.27 | 1016.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 1018.15 | 1020.27 | 1016.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1021.00 | 1020.42 | 1016.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 1019.45 | 1020.42 | 1016.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1020.00 | 1020.34 | 1016.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 1019.30 | 1020.34 | 1016.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1019.40 | 1020.15 | 1017.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 1019.40 | 1020.15 | 1017.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1021.00 | 1020.32 | 1017.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1020.00 | 1020.32 | 1017.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 994.70 | 1015.19 | 1015.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 964.50 | 995.93 | 1005.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1004.60 | 974.05 | 985.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1004.60 | 974.05 | 985.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1004.60 | 974.05 | 985.14 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1013.70 | 991.61 | 991.27 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 986.80 | 995.02 | 995.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 982.80 | 990.51 | 993.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 992.10 | 990.83 | 993.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 992.10 | 990.83 | 993.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 992.10 | 990.83 | 993.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 992.10 | 990.83 | 993.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 994.00 | 991.46 | 993.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 993.30 | 991.46 | 993.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 987.80 | 990.73 | 992.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 983.80 | 990.73 | 992.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 985.80 | 988.01 | 990.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 14:15:00 | 1004.40 | 991.65 | 991.90 | SL hit (close>static) qty=1.00 sl=998.50 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 1001.00 | 993.52 | 992.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1019.70 | 998.76 | 995.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1067.80 | 1072.57 | 1060.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:30:00 | 1071.10 | 1072.57 | 1060.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1064.20 | 1069.85 | 1062.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 1064.20 | 1069.85 | 1062.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1069.20 | 1069.72 | 1063.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 1070.60 | 1069.72 | 1063.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1074.60 | 1068.01 | 1063.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 1069.90 | 1069.43 | 1064.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1073.10 | 1070.09 | 1068.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1070.40 | 1073.91 | 1072.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 1069.80 | 1073.91 | 1072.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1069.90 | 1073.11 | 1072.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1069.90 | 1073.11 | 1072.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1067.10 | 1071.91 | 1071.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1067.10 | 1071.91 | 1071.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 1069.10 | 1071.35 | 1071.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1069.10 | 1071.35 | 1071.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1065.60 | 1070.20 | 1070.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1060.50 | 1060.37 | 1064.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:30:00 | 1059.70 | 1060.37 | 1064.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1058.40 | 1059.98 | 1064.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1053.30 | 1063.26 | 1063.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 1055.60 | 1059.94 | 1062.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 1053.30 | 1059.39 | 1061.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 1055.50 | 1058.86 | 1061.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1061.10 | 1059.15 | 1060.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 1061.50 | 1059.15 | 1060.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1065.50 | 1060.42 | 1061.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 1065.50 | 1060.42 | 1061.39 | SL hit (close>static) qty=1.00 sl=1064.80 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1075.60 | 1063.46 | 1062.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1081.00 | 1066.97 | 1064.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1085.20 | 1092.86 | 1084.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1085.20 | 1092.86 | 1084.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1085.20 | 1092.86 | 1084.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 1083.10 | 1092.86 | 1084.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1084.80 | 1091.25 | 1084.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1082.80 | 1091.25 | 1084.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1080.10 | 1089.02 | 1084.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 1080.10 | 1089.02 | 1084.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1078.30 | 1086.87 | 1083.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 1078.30 | 1086.87 | 1083.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1080.20 | 1085.08 | 1083.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1080.20 | 1085.08 | 1083.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1079.50 | 1083.96 | 1083.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1061.50 | 1083.96 | 1083.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1060.00 | 1079.17 | 1081.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1000.00 | 1046.61 | 1061.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1035.10 | 1021.92 | 1038.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 1035.10 | 1021.92 | 1038.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1037.60 | 1025.06 | 1038.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 1037.60 | 1025.06 | 1038.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1031.50 | 1026.35 | 1037.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1028.10 | 1027.53 | 1036.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1039.30 | 1029.88 | 1036.43 | SL hit (close>static) qty=1.00 sl=1037.60 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1046.60 | 1015.67 | 1014.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 1055.70 | 1023.68 | 1017.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1040.00 | 1047.30 | 1036.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 1040.00 | 1047.30 | 1036.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1033.90 | 1044.62 | 1036.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1033.70 | 1044.62 | 1036.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1029.80 | 1041.66 | 1035.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1029.80 | 1041.66 | 1035.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1038.00 | 1040.93 | 1035.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1014.70 | 1040.93 | 1035.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1016.00 | 1035.94 | 1033.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1011.20 | 1035.94 | 1033.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1035.90 | 1034.96 | 1033.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 1035.90 | 1034.96 | 1033.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 1030.70 | 1034.78 | 1033.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 1036.50 | 1034.78 | 1033.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1029.10 | 1033.65 | 1033.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 1019.10 | 1033.65 | 1033.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1009.70 | 1028.86 | 1031.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1001.40 | 1023.37 | 1028.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1007.50 | 995.06 | 1003.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1007.50 | 995.06 | 1003.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1007.50 | 995.06 | 1003.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1007.50 | 995.06 | 1003.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1005.90 | 997.23 | 1003.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1000.00 | 997.23 | 1003.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 1000.80 | 999.93 | 1003.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1000.00 | 999.93 | 1003.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1013.10 | 1003.84 | 1004.61 | SL hit (close>static) qty=1.00 sl=1012.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1012.30 | 1005.54 | 1005.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1020.40 | 1010.82 | 1007.97 | Break + close above crossover candle high |

### Cycle 66 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 980.80 | 1007.37 | 1007.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 967.20 | 999.33 | 1003.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 898.80 | 894.90 | 921.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:30:00 | 904.10 | 894.90 | 921.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 916.00 | 903.32 | 918.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 928.40 | 903.32 | 918.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 944.70 | 913.40 | 918.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 941.20 | 913.40 | 918.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 957.50 | 922.22 | 922.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 957.50 | 922.22 | 922.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 954.90 | 928.76 | 925.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 962.20 | 939.68 | 931.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 913.10 | 939.16 | 933.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 913.10 | 939.16 | 933.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 913.10 | 939.16 | 933.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 913.10 | 939.16 | 933.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 915.90 | 934.51 | 931.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 910.80 | 934.51 | 931.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 907.50 | 929.11 | 929.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 903.50 | 918.82 | 924.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 912.25 | 888.52 | 900.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 912.25 | 888.52 | 900.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 912.25 | 888.52 | 900.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 908.25 | 888.52 | 900.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 904.30 | 899.36 | 902.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 905.65 | 901.19 | 903.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 906.20 | 895.91 | 895.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 906.20 | 895.91 | 895.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 916.50 | 900.03 | 897.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 998.70 | 1004.22 | 979.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 998.70 | 1004.22 | 979.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 995.55 | 1014.68 | 1002.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 1010.00 | 1011.30 | 1003.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1024.25 | 1039.16 | 1039.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1024.25 | 1039.16 | 1039.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 1019.20 | 1035.17 | 1037.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1021.10 | 1009.20 | 1017.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1021.10 | 1009.20 | 1017.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 1009.20 | 1017.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1021.10 | 1009.20 | 1017.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1016.00 | 1010.56 | 1017.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 978.20 | 1010.56 | 1017.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 929.29 | 954.81 | 966.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 960.80 | 944.19 | 953.09 | SL hit (close>ema200) qty=0.50 sl=944.19 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 962.10 | 956.72 | 956.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 965.00 | 960.00 | 957.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1006.70 | 1007.77 | 994.14 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-30 10:15:00 | 640.50 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-30 12:30:00 | 641.75 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-30 13:45:00 | 642.30 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-02 12:45:00 | 641.95 | 2025-06-03 13:15:00 | 651.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-17 09:45:00 | 673.05 | 2025-06-24 09:15:00 | 676.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-04 11:15:00 | 675.80 | 2025-07-09 10:15:00 | 679.05 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-04 14:00:00 | 675.90 | 2025-07-09 10:15:00 | 679.05 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-04 15:00:00 | 675.80 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-07 09:15:00 | 675.20 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-08 11:30:00 | 669.65 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-08 14:00:00 | 668.35 | 2025-07-09 12:15:00 | 682.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-07-22 11:00:00 | 645.80 | 2025-07-25 09:15:00 | 613.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:30:00 | 645.95 | 2025-07-25 09:15:00 | 613.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:15:00 | 636.35 | 2025-07-25 11:15:00 | 604.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 645.80 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-07-22 11:30:00 | 645.95 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-07-24 10:15:00 | 636.35 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2025-08-12 09:30:00 | 614.80 | 2025-08-13 14:15:00 | 619.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-12 10:15:00 | 613.85 | 2025-08-13 14:15:00 | 619.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-08-13 09:45:00 | 614.85 | 2025-08-13 14:15:00 | 619.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-04 11:00:00 | 586.90 | 2025-09-05 10:15:00 | 591.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-09-04 13:15:00 | 586.40 | 2025-09-05 10:15:00 | 591.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-05 10:00:00 | 586.55 | 2025-09-05 10:15:00 | 591.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-09 12:45:00 | 594.30 | 2025-09-16 13:15:00 | 615.50 | STOP_HIT | 1.00 | 3.57% |
| BUY | retest2 | 2025-09-09 13:15:00 | 595.65 | 2025-09-16 13:15:00 | 615.50 | STOP_HIT | 1.00 | 3.33% |
| BUY | retest2 | 2025-09-09 14:00:00 | 594.30 | 2025-09-16 13:15:00 | 615.50 | STOP_HIT | 1.00 | 3.57% |
| BUY | retest2 | 2025-10-13 12:45:00 | 672.00 | 2025-10-16 10:15:00 | 664.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-13 15:00:00 | 672.95 | 2025-10-16 10:15:00 | 664.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-14 13:15:00 | 671.80 | 2025-10-16 10:15:00 | 664.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-28 11:30:00 | 719.45 | 2025-11-03 09:15:00 | 791.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-24 10:15:00 | 838.90 | 2025-12-01 15:15:00 | 851.95 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-11-25 11:15:00 | 835.20 | 2025-12-01 15:15:00 | 851.95 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2025-11-25 11:45:00 | 835.00 | 2025-12-01 15:15:00 | 851.95 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2025-12-30 09:15:00 | 969.00 | 2026-01-06 12:15:00 | 993.65 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2026-01-19 12:15:00 | 1012.40 | 2026-01-20 14:15:00 | 984.85 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-22 11:15:00 | 992.15 | 2026-01-22 12:15:00 | 1000.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-06 10:15:00 | 983.80 | 2026-02-06 14:15:00 | 1004.40 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-02-06 12:15:00 | 985.80 | 2026-02-06 14:15:00 | 1004.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-13 14:15:00 | 1070.60 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-16 09:15:00 | 1074.60 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-16 10:30:00 | 1069.90 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-02-17 14:30:00 | 1073.10 | 2026-02-19 12:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1053.30 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-02-24 11:00:00 | 1055.60 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1053.30 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-02-24 13:15:00 | 1055.50 | 2026-02-24 15:15:00 | 1065.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-02-25 09:15:00 | 1059.30 | 2026-02-25 09:15:00 | 1075.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1028.10 | 2026-03-05 14:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-06 13:45:00 | 1026.50 | 2026-03-09 09:15:00 | 975.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 1026.50 | 2026-03-09 15:15:00 | 989.90 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1000.00 | 2026-03-18 09:15:00 | 1013.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-17 13:30:00 | 1000.80 | 2026-03-18 09:15:00 | 1013.10 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-03-17 14:15:00 | 1000.00 | 2026-03-18 09:15:00 | 1013.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-01 10:15:00 | 908.25 | 2026-04-06 11:15:00 | 906.20 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2026-04-01 13:15:00 | 904.30 | 2026-04-06 11:15:00 | 906.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-04-01 14:15:00 | 905.65 | 2026-04-06 11:15:00 | 906.20 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-04-13 12:00:00 | 1010.00 | 2026-04-23 09:15:00 | 1024.25 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2026-04-27 09:15:00 | 978.20 | 2026-04-30 09:15:00 | 929.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 09:15:00 | 978.20 | 2026-05-04 09:15:00 | 960.80 | STOP_HIT | 0.50 | 1.78% |
