# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1044.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 52 |
| ALERT2 | 50 |
| ALERT2_SKIP | 28 |
| ALERT3 | 152 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 60 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 44
- **Target hits / Stop hits / Partials:** 3 / 58 / 0
- **Avg / median % per leg:** -0.14% / -0.85%
- **Sum % (uncompounded):** -8.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 17 | 38.6% | 3 | 41 | 0 | 0.29% | 12.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 44 | 17 | 38.6% | 3 | 41 | 0 | 0.29% | 12.7% |
| SELL (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -1.26% | -21.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.35% | -2.4% |
| SELL @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.19% | -19.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.35% | -2.4% |
| retest2 (combined) | 60 | 17 | 28.3% | 3 | 57 | 0 | -0.11% | -6.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 644.00 | 628.52 | 627.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 651.60 | 647.82 | 643.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 655.40 | 656.66 | 651.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 655.40 | 656.66 | 651.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 657.30 | 656.89 | 653.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 654.75 | 656.89 | 653.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 663.50 | 662.06 | 659.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 660.50 | 662.06 | 659.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 662.35 | 662.78 | 660.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 656.10 | 662.78 | 660.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 662.40 | 662.70 | 660.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 662.40 | 662.70 | 660.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 657.70 | 661.70 | 660.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 657.70 | 661.70 | 660.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 659.85 | 661.33 | 660.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 660.50 | 661.33 | 660.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 654.00 | 660.32 | 660.10 | SL hit (close<static) qty=1.00 sl=655.70 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 655.00 | 659.26 | 659.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 649.45 | 656.78 | 658.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 652.70 | 651.89 | 654.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:00:00 | 652.70 | 651.89 | 654.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 654.40 | 652.39 | 654.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 654.45 | 652.39 | 654.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 654.50 | 652.81 | 654.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 653.45 | 652.81 | 654.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 649.55 | 652.16 | 654.14 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 660.05 | 655.90 | 655.46 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 652.55 | 658.09 | 658.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 652.10 | 656.89 | 657.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 651.00 | 650.88 | 653.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 651.00 | 650.88 | 653.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 634.85 | 632.45 | 637.04 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 640.15 | 636.69 | 636.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 642.95 | 638.55 | 637.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 656.25 | 656.54 | 652.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:00:00 | 656.25 | 656.54 | 652.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 654.80 | 656.91 | 654.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 654.80 | 656.91 | 654.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 658.20 | 656.92 | 654.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 658.75 | 656.92 | 654.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 659.05 | 656.99 | 655.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 651.00 | 653.79 | 654.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 651.00 | 653.79 | 654.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 645.10 | 652.05 | 653.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 644.65 | 643.87 | 647.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:45:00 | 645.30 | 643.87 | 647.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 646.75 | 644.91 | 647.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 648.00 | 644.91 | 647.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 647.80 | 645.49 | 647.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 647.80 | 645.49 | 647.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 650.00 | 646.39 | 647.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 645.75 | 646.85 | 647.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 646.85 | 644.41 | 645.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 653.85 | 644.85 | 644.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 653.85 | 644.85 | 644.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 657.40 | 648.11 | 646.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 692.35 | 693.41 | 686.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 11:15:00 | 692.30 | 693.41 | 686.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 685.60 | 691.76 | 688.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 685.60 | 691.76 | 688.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 687.10 | 690.83 | 688.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 686.85 | 690.83 | 688.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 695.20 | 691.70 | 689.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 695.55 | 692.27 | 689.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 699.10 | 693.36 | 690.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:15:00 | 696.65 | 693.77 | 691.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 698.45 | 694.03 | 691.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 695.80 | 696.90 | 693.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 695.70 | 696.90 | 693.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 696.75 | 699.34 | 697.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 696.75 | 699.34 | 697.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 692.35 | 697.94 | 696.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 692.35 | 697.94 | 696.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 692.35 | 696.82 | 696.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 692.95 | 696.82 | 696.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 693.90 | 695.75 | 695.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 693.90 | 695.75 | 695.80 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 699.50 | 696.10 | 695.92 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 691.35 | 695.15 | 695.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 689.00 | 693.35 | 694.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 676.45 | 675.56 | 679.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 676.45 | 675.56 | 679.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 675.05 | 675.46 | 679.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 676.20 | 675.46 | 679.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 679.25 | 674.15 | 676.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 679.25 | 674.15 | 676.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 672.20 | 673.76 | 675.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 666.95 | 673.76 | 675.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 670.30 | 670.01 | 671.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 667.00 | 670.40 | 671.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 669.00 | 668.80 | 669.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 673.80 | 669.80 | 669.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 673.45 | 669.80 | 669.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 674.75 | 670.79 | 670.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 674.75 | 670.79 | 670.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 681.05 | 676.09 | 674.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 693.35 | 695.58 | 692.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 693.35 | 695.58 | 692.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 692.55 | 694.98 | 692.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 692.55 | 694.98 | 692.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 690.90 | 694.16 | 692.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 691.30 | 694.16 | 692.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 691.70 | 693.67 | 692.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:00:00 | 693.90 | 693.38 | 692.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 685.25 | 691.87 | 691.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 685.25 | 691.87 | 691.89 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 693.05 | 690.80 | 690.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 695.00 | 691.64 | 690.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 691.45 | 691.60 | 690.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 691.45 | 691.60 | 690.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 691.45 | 691.60 | 690.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:15:00 | 687.90 | 691.60 | 690.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 690.60 | 691.40 | 690.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 686.90 | 691.40 | 690.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 687.85 | 690.69 | 690.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 687.85 | 690.69 | 690.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 685.20 | 689.59 | 690.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 678.65 | 686.97 | 688.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 687.20 | 686.35 | 688.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 687.20 | 686.35 | 688.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 687.20 | 686.35 | 688.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 690.40 | 686.35 | 688.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 689.40 | 686.96 | 688.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 689.40 | 686.96 | 688.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 690.10 | 687.59 | 688.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 690.10 | 687.59 | 688.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 685.80 | 679.03 | 681.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 685.80 | 679.03 | 681.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 686.30 | 680.48 | 682.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 686.30 | 680.48 | 682.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 687.10 | 683.25 | 683.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 688.50 | 684.95 | 684.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 12:15:00 | 682.80 | 685.41 | 684.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 12:15:00 | 682.80 | 685.41 | 684.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 682.80 | 685.41 | 684.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 682.80 | 685.41 | 684.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 683.30 | 684.99 | 684.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 685.30 | 685.41 | 684.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:00:00 | 687.10 | 685.41 | 684.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 679.50 | 685.19 | 685.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 679.50 | 685.19 | 685.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 676.15 | 681.91 | 683.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 686.50 | 682.67 | 683.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 686.50 | 682.67 | 683.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 686.50 | 682.67 | 683.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 686.50 | 682.67 | 683.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 689.65 | 684.07 | 684.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 682.70 | 684.07 | 684.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 697.10 | 674.17 | 674.22 | SL hit (close>static) qty=1.00 sl=689.90 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 702.55 | 679.84 | 676.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 712.00 | 697.19 | 691.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 10:15:00 | 707.20 | 709.02 | 702.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:00:00 | 707.20 | 709.02 | 702.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 700.35 | 706.07 | 703.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 698.95 | 706.07 | 703.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 702.25 | 705.31 | 703.50 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 701.00 | 702.39 | 702.54 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 706.15 | 703.27 | 702.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 706.95 | 704.00 | 703.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 705.65 | 705.80 | 704.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 705.65 | 705.80 | 704.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 705.65 | 705.80 | 704.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 704.85 | 705.80 | 704.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 705.65 | 705.77 | 704.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 705.65 | 705.77 | 704.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 704.75 | 706.16 | 705.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 704.75 | 706.16 | 705.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 702.65 | 705.46 | 705.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 715.45 | 705.46 | 705.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:00:00 | 705.40 | 709.57 | 709.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 704.80 | 708.62 | 708.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 704.80 | 708.62 | 708.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 704.00 | 707.19 | 707.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 706.05 | 705.67 | 706.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 11:15:00 | 706.05 | 705.67 | 706.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 706.05 | 705.67 | 706.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:30:00 | 706.05 | 705.67 | 706.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 705.60 | 702.93 | 704.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 705.60 | 702.93 | 704.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 706.00 | 703.55 | 704.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 706.00 | 703.55 | 704.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 705.80 | 704.00 | 705.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 706.65 | 704.00 | 705.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 702.80 | 703.77 | 704.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 706.00 | 703.77 | 704.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 709.00 | 704.85 | 705.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 709.00 | 704.85 | 705.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 710.10 | 705.90 | 705.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 713.25 | 707.37 | 706.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 738.70 | 738.89 | 732.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:15:00 | 738.00 | 738.89 | 732.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 741.95 | 743.54 | 740.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 741.95 | 743.54 | 740.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 738.70 | 742.58 | 740.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 738.70 | 742.58 | 740.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 738.50 | 741.76 | 739.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 742.75 | 741.99 | 740.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:15:00 | 742.00 | 741.76 | 740.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 740.05 | 742.26 | 742.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 740.05 | 742.26 | 742.38 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 750.00 | 743.81 | 743.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 13:15:00 | 758.95 | 748.68 | 745.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 753.50 | 755.00 | 751.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 753.50 | 755.00 | 751.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 752.75 | 754.15 | 751.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 753.95 | 754.15 | 751.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 753.50 | 754.02 | 751.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 750.40 | 754.02 | 751.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 753.70 | 753.95 | 752.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:30:00 | 756.00 | 754.43 | 752.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 756.00 | 754.43 | 752.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 756.20 | 754.78 | 752.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 751.15 | 753.98 | 753.03 | SL hit (close<static) qty=1.00 sl=751.75 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 750.05 | 752.19 | 752.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 746.75 | 750.67 | 751.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 750.50 | 748.53 | 749.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 750.50 | 748.53 | 749.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 750.50 | 748.53 | 749.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 750.50 | 748.53 | 749.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 750.00 | 748.82 | 749.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 746.00 | 748.82 | 749.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 743.50 | 747.76 | 749.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 741.90 | 746.17 | 748.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 750.00 | 747.78 | 747.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 750.00 | 747.78 | 747.71 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 745.55 | 747.33 | 747.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 741.55 | 746.22 | 746.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 750.80 | 745.98 | 746.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 750.80 | 745.98 | 746.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 750.80 | 745.98 | 746.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 750.80 | 745.98 | 746.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 749.70 | 746.73 | 746.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:00:00 | 745.80 | 746.54 | 746.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 752.95 | 745.86 | 745.99 | SL hit (close>static) qty=1.00 sl=752.25 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 751.45 | 746.98 | 746.48 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 740.15 | 745.95 | 746.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 735.80 | 743.92 | 745.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 744.10 | 741.72 | 743.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 744.10 | 741.72 | 743.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 744.10 | 741.72 | 743.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:30:00 | 747.00 | 741.72 | 743.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 742.10 | 741.79 | 743.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 750.70 | 741.79 | 743.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 751.85 | 743.81 | 744.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 751.70 | 743.81 | 744.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 10:15:00 | 754.05 | 745.85 | 745.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 756.35 | 749.13 | 746.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 757.00 | 760.83 | 756.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 757.00 | 760.83 | 756.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 757.00 | 760.83 | 756.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 756.20 | 760.83 | 756.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 761.35 | 760.94 | 757.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 756.15 | 760.94 | 757.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 778.15 | 777.25 | 774.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 771.80 | 777.25 | 774.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 771.00 | 776.00 | 773.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 771.55 | 776.00 | 773.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 769.95 | 774.79 | 773.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 768.35 | 774.79 | 773.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 767.50 | 771.95 | 772.46 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 775.75 | 772.12 | 772.10 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 765.95 | 772.40 | 772.60 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 772.25 | 772.17 | 772.17 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 769.70 | 771.68 | 771.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 764.80 | 769.93 | 770.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 764.05 | 762.39 | 765.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 764.05 | 762.39 | 765.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 764.05 | 762.39 | 765.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 766.95 | 762.39 | 765.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 765.10 | 762.93 | 765.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 765.10 | 762.93 | 765.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 765.85 | 763.52 | 765.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 765.85 | 763.52 | 765.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 764.35 | 763.68 | 765.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 763.15 | 763.71 | 765.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 768.95 | 764.96 | 765.58 | SL hit (close>static) qty=1.00 sl=766.80 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 775.50 | 767.07 | 766.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 778.05 | 771.98 | 769.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 775.85 | 775.85 | 772.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 11:00:00 | 775.85 | 775.85 | 772.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 767.25 | 774.00 | 771.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 767.25 | 774.00 | 771.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 769.95 | 773.19 | 771.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 772.50 | 773.19 | 771.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 849.75 | 838.86 | 824.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 15:15:00 | 845.50 | 850.18 | 850.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 842.95 | 848.73 | 849.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 795.30 | 793.21 | 808.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 796.75 | 793.21 | 808.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 793.80 | 787.93 | 792.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 793.80 | 787.93 | 792.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 792.65 | 788.87 | 792.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 796.80 | 788.87 | 792.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 803.85 | 791.87 | 793.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 803.85 | 791.87 | 793.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 803.90 | 795.90 | 795.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 804.70 | 797.88 | 796.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 807.05 | 807.66 | 803.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 808.55 | 807.84 | 803.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 808.55 | 807.84 | 803.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 803.75 | 807.84 | 803.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 804.60 | 807.19 | 803.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 803.70 | 807.19 | 803.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 799.70 | 805.69 | 803.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 799.70 | 805.69 | 803.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 800.15 | 804.58 | 803.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 799.70 | 804.58 | 803.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 803.00 | 804.17 | 803.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 804.90 | 804.17 | 803.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 803.10 | 803.96 | 803.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 807.35 | 804.57 | 803.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 806.90 | 805.03 | 803.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 789.50 | 802.40 | 802.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 789.50 | 802.40 | 802.92 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 799.50 | 797.42 | 797.33 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 784.70 | 795.30 | 796.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 775.95 | 780.50 | 785.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 785.75 | 780.48 | 784.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 785.75 | 780.48 | 784.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 785.75 | 780.48 | 784.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 784.75 | 780.48 | 784.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 784.80 | 781.34 | 784.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 783.85 | 781.84 | 784.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 782.25 | 782.16 | 784.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 794.00 | 786.64 | 785.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 794.00 | 786.64 | 785.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 799.80 | 789.27 | 787.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 807.65 | 808.09 | 803.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 807.65 | 808.09 | 803.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 806.50 | 810.14 | 807.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 806.50 | 810.14 | 807.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 805.95 | 809.30 | 807.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 805.95 | 809.30 | 807.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 805.80 | 808.60 | 807.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 805.80 | 808.60 | 807.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 803.75 | 807.63 | 806.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 802.75 | 807.63 | 806.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 806.95 | 807.50 | 806.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 808.05 | 807.61 | 807.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 808.80 | 807.08 | 806.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:30:00 | 808.10 | 808.22 | 807.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:00:00 | 811.05 | 819.66 | 819.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 817.50 | 819.23 | 819.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 812.00 | 819.23 | 819.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 815.15 | 818.42 | 818.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 815.15 | 818.42 | 818.75 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 825.20 | 818.78 | 818.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 829.50 | 822.00 | 820.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 822.15 | 822.18 | 820.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 822.15 | 822.18 | 820.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 822.15 | 822.18 | 820.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 820.70 | 822.18 | 820.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 821.70 | 822.08 | 820.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 827.60 | 822.08 | 820.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 824.50 | 823.94 | 822.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 13:15:00 | 832.60 | 838.89 | 839.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 832.60 | 838.89 | 839.21 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 845.50 | 839.57 | 839.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 853.30 | 847.10 | 843.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 10:15:00 | 849.65 | 854.04 | 850.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 849.65 | 854.04 | 850.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 849.65 | 854.04 | 850.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 849.65 | 854.04 | 850.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 851.40 | 853.51 | 850.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 853.75 | 852.02 | 850.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 15:15:00 | 865.35 | 867.65 | 867.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 865.35 | 867.65 | 867.95 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 875.35 | 869.19 | 868.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 880.55 | 872.52 | 870.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 942.80 | 944.84 | 930.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 942.80 | 944.84 | 930.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 939.50 | 942.09 | 936.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 939.90 | 942.09 | 936.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 918.75 | 936.74 | 935.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 918.75 | 936.74 | 935.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 905.90 | 930.57 | 932.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 903.10 | 925.08 | 929.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 904.00 | 902.79 | 911.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 902.00 | 902.79 | 911.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 902.45 | 902.72 | 910.28 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 921.35 | 912.76 | 912.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 927.50 | 915.71 | 913.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 946.70 | 948.39 | 939.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 946.70 | 948.39 | 939.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 939.65 | 946.20 | 939.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 939.65 | 946.20 | 939.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 932.05 | 943.37 | 939.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 932.05 | 943.37 | 939.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 934.50 | 941.59 | 938.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 932.15 | 941.59 | 938.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 941.45 | 939.15 | 938.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 945.60 | 940.44 | 938.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 13:15:00 | 933.80 | 938.49 | 938.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 933.80 | 938.49 | 938.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 927.10 | 936.21 | 937.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 10:15:00 | 934.60 | 934.37 | 936.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 11:00:00 | 934.60 | 934.37 | 936.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 936.30 | 934.76 | 936.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 936.30 | 934.76 | 936.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 940.40 | 935.89 | 936.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 940.40 | 935.89 | 936.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 935.80 | 935.87 | 936.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 938.85 | 935.87 | 936.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 938.45 | 936.39 | 936.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 939.55 | 936.39 | 936.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 940.00 | 937.11 | 937.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 943.30 | 938.35 | 937.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 10:15:00 | 936.30 | 937.94 | 937.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 10:15:00 | 936.30 | 937.94 | 937.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 936.30 | 937.94 | 937.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 936.30 | 937.94 | 937.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 939.60 | 938.27 | 937.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:15:00 | 942.80 | 938.27 | 937.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 13:45:00 | 943.95 | 940.05 | 938.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 957.40 | 986.16 | 988.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 957.40 | 986.16 | 988.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 921.40 | 963.65 | 976.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 920.00 | 916.92 | 935.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 920.00 | 916.92 | 935.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 936.10 | 922.86 | 934.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 952.15 | 922.86 | 934.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 953.65 | 929.02 | 936.54 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 968.80 | 944.71 | 942.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 15:15:00 | 970.00 | 961.17 | 954.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 932.00 | 955.33 | 952.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 932.00 | 955.33 | 952.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 932.00 | 955.33 | 952.59 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 931.75 | 950.62 | 950.70 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 967.45 | 946.71 | 944.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 970.40 | 962.32 | 955.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 953.90 | 964.20 | 959.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 953.90 | 964.20 | 959.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 953.90 | 964.20 | 959.48 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 907.85 | 952.01 | 956.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 09:15:00 | 886.00 | 903.94 | 917.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 897.30 | 894.06 | 904.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 897.30 | 894.06 | 904.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 897.30 | 894.06 | 904.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 903.25 | 894.06 | 904.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 901.85 | 897.28 | 901.22 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 914.05 | 905.17 | 904.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 923.95 | 910.42 | 906.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 924.80 | 927.15 | 918.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:00:00 | 924.80 | 927.15 | 918.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 914.50 | 924.62 | 918.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 911.70 | 924.62 | 918.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 908.50 | 921.40 | 917.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 908.50 | 921.40 | 917.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 910.65 | 919.25 | 917.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 916.55 | 918.71 | 916.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 918.40 | 917.87 | 916.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 915.75 | 917.79 | 917.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 916.80 | 917.60 | 916.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 917.45 | 917.57 | 917.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 914.00 | 917.57 | 917.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 923.15 | 918.68 | 917.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 925.30 | 918.68 | 917.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 925.40 | 931.92 | 931.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 925.40 | 930.61 | 931.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 925.40 | 930.61 | 931.21 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 936.95 | 931.88 | 931.73 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 925.70 | 930.64 | 931.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 923.10 | 929.14 | 930.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 934.40 | 929.69 | 930.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 934.40 | 929.69 | 930.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 934.40 | 929.69 | 930.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 934.40 | 929.69 | 930.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 941.15 | 931.98 | 931.41 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 924.10 | 930.13 | 930.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 915.65 | 927.23 | 929.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 931.05 | 927.43 | 929.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 13:15:00 | 931.05 | 927.43 | 929.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 931.05 | 927.43 | 929.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 931.05 | 927.43 | 929.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 920.95 | 926.14 | 928.34 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 970.00 | 934.33 | 931.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 10:15:00 | 981.90 | 943.84 | 936.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 14:15:00 | 954.35 | 958.06 | 946.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 954.35 | 958.06 | 946.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 954.35 | 958.06 | 946.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 954.35 | 958.06 | 946.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 963.10 | 958.85 | 948.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 10:15:00 | 966.40 | 958.85 | 948.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:45:00 | 964.95 | 960.97 | 952.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 14:00:00 | 964.70 | 961.71 | 953.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 940.60 | 957.29 | 953.50 | SL hit (close<static) qty=1.00 sl=943.15 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 941.00 | 949.74 | 950.46 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 955.55 | 948.15 | 947.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 965.15 | 953.20 | 950.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 954.60 | 956.40 | 952.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 954.60 | 956.40 | 952.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 954.60 | 956.40 | 952.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 954.60 | 956.40 | 952.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 936.35 | 953.42 | 952.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 936.35 | 953.42 | 952.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 945.00 | 951.74 | 951.94 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 953.60 | 952.09 | 952.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 966.05 | 954.88 | 953.34 | Break + close above crossover candle high |

### Cycle 68 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 915.90 | 951.30 | 952.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 913.50 | 930.33 | 940.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 926.45 | 923.84 | 934.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 926.45 | 923.84 | 934.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 926.45 | 923.84 | 934.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 930.65 | 923.84 | 934.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 937.95 | 927.46 | 934.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:30:00 | 940.45 | 927.46 | 934.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 926.05 | 927.18 | 933.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 13:15:00 | 923.50 | 927.18 | 933.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 15:00:00 | 920.00 | 925.07 | 931.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 935.00 | 933.11 | 933.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 935.00 | 933.11 | 933.00 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 925.10 | 931.50 | 932.28 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 934.20 | 932.70 | 932.53 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 922.80 | 930.58 | 931.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 906.65 | 921.58 | 926.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 861.00 | 852.29 | 866.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 861.80 | 852.29 | 866.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 869.15 | 856.48 | 863.97 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 11:15:00 | 867.00 | 866.49 | 866.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 907.50 | 875.27 | 870.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 889.75 | 901.03 | 893.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 889.75 | 901.03 | 893.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 889.75 | 901.03 | 893.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 889.75 | 901.03 | 893.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 890.15 | 898.85 | 893.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 887.80 | 898.85 | 893.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 910.40 | 900.31 | 894.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 916.90 | 900.31 | 894.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 916.60 | 906.40 | 898.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1008.59 | 985.17 | 978.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 1015.55 | 1022.17 | 1022.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 1013.10 | 1020.35 | 1021.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 1022.75 | 1020.51 | 1021.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 1022.75 | 1020.51 | 1021.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1022.75 | 1020.51 | 1021.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1021.30 | 1020.51 | 1021.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1022.20 | 1020.85 | 1021.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:15:00 | 1022.10 | 1020.85 | 1021.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1021.65 | 1021.01 | 1021.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:30:00 | 1019.75 | 1021.06 | 1021.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 14:30:00 | 1019.65 | 1021.26 | 1021.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 1031.60 | 1023.33 | 1022.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 1031.60 | 1023.33 | 1022.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 1038.00 | 1028.78 | 1025.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 1034.15 | 1038.75 | 1034.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 1034.15 | 1038.75 | 1034.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1034.15 | 1038.75 | 1034.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:15:00 | 1029.00 | 1038.75 | 1034.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1032.75 | 1037.55 | 1034.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 1028.00 | 1037.55 | 1034.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1068.00 | 1068.81 | 1061.27 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1045.05 | 1060.91 | 1061.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 1038.60 | 1050.57 | 1056.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 15:15:00 | 1043.90 | 1043.86 | 1048.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1033.40 | 1043.86 | 1048.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1041.40 | 1043.37 | 1047.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 1041.40 | 1043.37 | 1047.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1047.20 | 1044.13 | 1047.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 1047.20 | 1044.13 | 1047.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1045.00 | 1044.31 | 1047.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:15:00 | 1049.40 | 1044.31 | 1047.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1057.70 | 1046.99 | 1048.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1057.70 | 1046.99 | 1048.53 | SL hit (close>ema400) qty=1.00 sl=1048.53 alert=retest1 |

### Cycle 77 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 1055.30 | 1049.79 | 1049.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 1057.00 | 1051.23 | 1050.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 14:15:00 | 1044.00 | 1053.84 | 1052.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1044.00 | 1053.84 | 1052.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1044.00 | 1053.84 | 1052.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 1044.00 | 1053.84 | 1052.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1044.20 | 1051.91 | 1051.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1052.30 | 1051.91 | 1051.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1053.60 | 1052.18 | 1051.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 1059.20 | 1053.59 | 1052.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 1057.00 | 1054.55 | 1053.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 1043.00 | 1052.63 | 1052.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1043.00 | 1052.63 | 1052.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 1039.60 | 1047.61 | 1050.16 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 13:15:00 | 660.50 | 2025-05-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-12 10:15:00 | 658.75 | 2025-06-12 15:15:00 | 651.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-06-12 11:15:00 | 659.05 | 2025-06-12 15:15:00 | 651.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-06-17 11:45:00 | 645.75 | 2025-06-20 10:15:00 | 653.85 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-18 10:45:00 | 646.85 | 2025-06-20 10:15:00 | 653.85 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-01 12:30:00 | 695.55 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-07-02 09:15:00 | 699.10 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-02 10:15:00 | 696.65 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-02 11:15:00 | 698.45 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-14 11:15:00 | 666.95 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-15 13:30:00 | 670.30 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-16 09:15:00 | 667.00 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-07-17 10:30:00 | 669.00 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-25 15:00:00 | 693.90 | 2025-07-28 09:15:00 | 685.25 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-08-05 14:30:00 | 685.30 | 2025-08-07 09:15:00 | 679.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-08-05 15:00:00 | 687.10 | 2025-08-07 09:15:00 | 679.50 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-08 09:15:00 | 682.70 | 2025-08-13 09:15:00 | 697.10 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-08-25 09:15:00 | 715.45 | 2025-08-26 13:15:00 | 704.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-08-26 13:00:00 | 705.40 | 2025-08-26 13:15:00 | 704.80 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-09-09 09:30:00 | 742.75 | 2025-09-11 12:15:00 | 740.05 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-09-09 12:15:00 | 742.00 | 2025-09-11 12:15:00 | 740.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-09-16 13:30:00 | 756.00 | 2025-09-17 10:15:00 | 751.15 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-16 14:15:00 | 756.00 | 2025-09-17 10:15:00 | 751.15 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-16 15:00:00 | 756.20 | 2025-09-17 10:15:00 | 751.15 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-17 13:15:00 | 756.25 | 2025-09-17 13:15:00 | 748.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-19 10:45:00 | 741.90 | 2025-09-22 13:15:00 | 750.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-23 15:00:00 | 745.80 | 2025-09-25 09:15:00 | 752.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-15 14:45:00 | 763.15 | 2025-10-16 09:15:00 | 768.95 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-17 14:15:00 | 772.50 | 2025-10-28 09:15:00 | 849.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-17 13:30:00 | 807.35 | 2025-11-18 09:15:00 | 789.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-11-17 15:00:00 | 806.90 | 2025-11-18 09:15:00 | 789.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-11-25 11:30:00 | 783.85 | 2025-11-26 09:15:00 | 794.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-25 12:45:00 | 782.25 | 2025-11-26 09:15:00 | 794.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-12-02 15:00:00 | 808.05 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-12-03 10:30:00 | 808.80 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-12-03 11:30:00 | 808.10 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-12-09 10:00:00 | 811.05 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-12-11 09:15:00 | 827.60 | 2025-12-16 13:15:00 | 832.60 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-12-11 15:00:00 | 824.50 | 2025-12-16 13:15:00 | 832.60 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-12-19 15:15:00 | 853.75 | 2025-12-29 15:15:00 | 865.35 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2026-01-19 12:00:00 | 945.60 | 2026-01-20 13:15:00 | 933.80 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-01-22 12:15:00 | 942.80 | 2026-01-30 12:15:00 | 957.40 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2026-01-22 13:45:00 | 943.95 | 2026-01-30 12:15:00 | 957.40 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2026-02-23 15:00:00 | 916.55 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2026-02-24 09:15:00 | 918.40 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2026-02-24 12:00:00 | 915.75 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2026-02-24 13:00:00 | 916.80 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2026-02-24 15:15:00 | 925.30 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-02-27 15:15:00 | 925.40 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-03-06 10:15:00 | 966.40 | 2026-03-09 09:15:00 | 940.60 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-03-06 12:45:00 | 964.95 | 2026-03-09 09:15:00 | 940.60 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-03-06 14:00:00 | 964.70 | 2026-03-09 09:15:00 | 940.60 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-03-16 13:15:00 | 923.50 | 2026-03-17 15:15:00 | 935.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-03-16 15:00:00 | 920.00 | 2026-03-17 15:15:00 | 935.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-02 13:15:00 | 916.90 | 2026-04-15 09:15:00 | 1008.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 916.60 | 2026-04-15 09:15:00 | 1008.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-21 13:30:00 | 1019.75 | 2026-04-22 09:15:00 | 1031.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-21 14:30:00 | 1019.65 | 2026-04-22 09:15:00 | 1031.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2026-05-05 09:15:00 | 1033.40 | 2026-05-05 12:15:00 | 1057.70 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-05-07 12:00:00 | 1059.20 | 2026-05-08 09:15:00 | 1043.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-05-07 15:15:00 | 1057.00 | 2026-05-08 09:15:00 | 1043.00 | STOP_HIT | 1.00 | -1.32% |
