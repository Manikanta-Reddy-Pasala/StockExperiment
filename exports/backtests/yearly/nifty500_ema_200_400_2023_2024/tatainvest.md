# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 719.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 62 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 50 |
| PARTIAL | 4 |
| TARGET_HIT | 12 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 34
- **Target hits / Stop hits / Partials:** 12 / 38 / 4
- **Avg / median % per leg:** 1.23% / -0.82%
- **Sum % (uncompounded):** 66.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 12 | 29.3% | 12 | 29 | 0 | 1.48% | 60.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 41 | 12 | 29.3% | 12 | 29 | 0 | 1.48% | 60.7% |
| SELL (all) | 13 | 8 | 61.5% | 0 | 9 | 4 | 0.44% | 5.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 0 | 9 | 4 | 0.44% | 5.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 20 | 37.0% | 12 | 38 | 4 | 1.23% | 66.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 641.47 | 651.05 | 651.10 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 13:15:00 | 673.10 | 651.19 | 651.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 695.30 | 652.06 | 651.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 11:15:00 | 659.00 | 661.15 | 656.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 12:00:00 | 659.00 | 661.15 | 656.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 656.11 | 661.08 | 656.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 656.11 | 661.08 | 656.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 651.07 | 660.98 | 656.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 651.07 | 660.98 | 656.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 651.90 | 660.89 | 656.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 655.68 | 660.82 | 656.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 12:15:00 | 647.62 | 660.50 | 656.73 | SL hit (close<static) qty=1.00 sl=649.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 643.50 | 654.90 | 654.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 633.01 | 654.68 | 654.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 630.51 | 622.47 | 633.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 630.51 | 622.47 | 633.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 630.51 | 622.47 | 633.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 630.51 | 622.47 | 633.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 632.96 | 622.57 | 633.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 632.96 | 622.57 | 633.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 664.60 | 622.70 | 632.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 664.60 | 622.70 | 632.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 669.63 | 623.16 | 632.67 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 715.80 | 641.64 | 641.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 753.34 | 642.76 | 642.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 678.84 | 681.00 | 666.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:00:00 | 678.84 | 681.00 | 666.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 664.50 | 680.60 | 666.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 664.50 | 680.60 | 666.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 662.01 | 680.42 | 666.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 658.14 | 680.42 | 666.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 662.10 | 680.08 | 666.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 662.10 | 680.08 | 666.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 667.50 | 679.96 | 666.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:30:00 | 661.97 | 679.96 | 666.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 669.10 | 679.85 | 666.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:45:00 | 679.53 | 679.34 | 666.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 682.34 | 683.53 | 671.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 11:00:00 | 681.70 | 683.31 | 671.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 13:00:00 | 681.14 | 683.25 | 671.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 670.98 | 683.06 | 671.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 670.98 | 683.06 | 671.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 672.19 | 682.96 | 671.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:15:00 | 669.01 | 682.96 | 671.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 671.21 | 682.84 | 671.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:30:00 | 671.18 | 682.84 | 671.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 669.10 | 682.70 | 671.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 669.10 | 682.70 | 671.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 666.82 | 682.54 | 671.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 666.82 | 682.54 | 671.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 675.22 | 682.07 | 671.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 661.88 | 681.53 | 671.69 | SL hit (close<static) qty=1.00 sl=663.25 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 13:15:00 | 650.00 | 673.75 | 673.79 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 689.62 | 673.44 | 673.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 696.00 | 677.50 | 675.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 679.20 | 679.91 | 676.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 10:15:00 | 679.20 | 679.91 | 676.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 679.20 | 679.91 | 676.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 684.00 | 679.91 | 676.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 677.30 | 680.71 | 677.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 676.50 | 680.71 | 677.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 673.82 | 680.64 | 677.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 673.82 | 680.64 | 677.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 673.87 | 680.58 | 677.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 673.87 | 680.58 | 677.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 680.20 | 677.81 | 676.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 692.00 | 677.81 | 676.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:45:00 | 682.50 | 679.10 | 677.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 674.37 | 679.00 | 677.28 | SL hit (close<static) qty=1.00 sl=675.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 645.80 | 676.42 | 676.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 638.60 | 675.40 | 675.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 594.75 | 588.77 | 617.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 594.75 | 588.77 | 617.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 594.75 | 588.77 | 617.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 608.53 | 588.77 | 617.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 634.95 | 589.81 | 617.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:30:00 | 575.69 | 591.53 | 616.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 12:30:00 | 581.00 | 591.18 | 615.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 580.83 | 592.91 | 614.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 14:00:00 | 580.62 | 592.02 | 613.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 617.50 | 592.41 | 613.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 595.12 | 592.41 | 613.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 582.91 | 592.32 | 613.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-22 11:15:00 | 630.20 | 619.07 | 619.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 630.20 | 619.07 | 619.03 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 10:15:00 | 611.85 | 619.04 | 619.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 608.50 | 618.73 | 618.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 607.50 | 607.14 | 612.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-13 09:30:00 | 605.20 | 607.14 | 612.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 612.35 | 607.15 | 612.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 613.00 | 607.15 | 612.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 611.85 | 607.20 | 612.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 14:00:00 | 610.95 | 607.32 | 612.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 615.75 | 607.50 | 612.10 | SL hit (close>static) qty=1.00 sl=613.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 630.00 | 615.39 | 615.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 635.85 | 615.60 | 615.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 654.50 | 657.22 | 641.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:00:00 | 654.50 | 657.22 | 641.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 641.10 | 656.41 | 641.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 647.95 | 656.41 | 641.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 651.80 | 656.37 | 641.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 654.00 | 656.37 | 641.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:30:00 | 654.70 | 656.35 | 642.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 653.50 | 664.47 | 655.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 655.00 | 664.42 | 655.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 678.45 | 667.02 | 658.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 688.50 | 668.19 | 659.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 686.75 | 668.19 | 659.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 687.80 | 668.58 | 660.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 686.75 | 668.76 | 660.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-05 09:15:00 | 719.40 | 672.45 | 663.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 758.25 | 788.33 | 788.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 748.00 | 787.64 | 788.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 654.55 | 648.96 | 682.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 655.75 | 648.96 | 682.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.90 | 649.48 | 682.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 673.75 | 649.48 | 682.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 724.20 | 647.89 | 674.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 724.20 | 647.89 | 674.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 714.85 | 648.56 | 674.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 702.50 | 649.00 | 674.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 705.40 | 649.56 | 674.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 707.15 | 650.11 | 674.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 705.80 | 654.57 | 676.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 667.38 | 657.39 | 676.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 667.40 | 657.39 | 676.67 | SL hit (close>static) qty=0.50 sl=657.39 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 729.50 | 650.84 | 650.72 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-28 09:30:00 | 655.68 | 2024-06-28 12:15:00 | 647.62 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-07-01 09:15:00 | 661.62 | 2024-07-10 09:15:00 | 645.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-07-08 09:45:00 | 653.71 | 2024-07-10 09:15:00 | 645.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-07-08 10:15:00 | 653.04 | 2024-07-10 09:15:00 | 645.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-07-12 11:30:00 | 664.41 | 2024-07-15 12:15:00 | 648.30 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-07-12 13:15:00 | 653.63 | 2024-07-15 12:15:00 | 648.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-12 15:15:00 | 654.50 | 2024-07-15 12:15:00 | 648.30 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-15 11:00:00 | 654.01 | 2024-07-15 12:15:00 | 648.30 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-09-23 09:45:00 | 679.53 | 2024-10-04 14:15:00 | 661.88 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-09-30 09:30:00 | 682.34 | 2024-10-04 14:15:00 | 661.88 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-10-01 11:00:00 | 681.70 | 2024-10-04 14:15:00 | 661.88 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-10-01 13:00:00 | 681.14 | 2024-10-04 14:15:00 | 661.88 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-10-10 09:30:00 | 711.17 | 2024-10-21 10:15:00 | 746.02 | TARGET_HIT | 1.00 | 4.90% |
| BUY | retest2 | 2024-10-18 09:30:00 | 678.20 | 2024-10-21 10:15:00 | 745.88 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2024-10-18 10:00:00 | 678.07 | 2024-10-24 14:15:00 | 671.68 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-10-23 09:45:00 | 681.07 | 2024-10-24 14:15:00 | 671.68 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-10-24 09:15:00 | 680.12 | 2024-10-25 09:15:00 | 652.63 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2024-10-24 10:15:00 | 682.20 | 2024-10-25 09:15:00 | 652.63 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2024-10-31 09:30:00 | 684.78 | 2024-11-04 09:15:00 | 668.07 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-10-31 11:15:00 | 680.49 | 2024-11-04 09:15:00 | 668.07 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-11-07 09:15:00 | 677.97 | 2024-11-07 12:15:00 | 672.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-11-08 13:15:00 | 688.35 | 2024-11-13 09:15:00 | 670.04 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-11-11 09:15:00 | 683.90 | 2024-11-13 09:15:00 | 670.04 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-11-18 10:00:00 | 677.53 | 2024-11-18 12:15:00 | 672.90 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-12-26 09:15:00 | 692.00 | 2024-12-30 14:15:00 | 674.37 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-12-30 11:45:00 | 682.50 | 2024-12-30 14:15:00 | 674.37 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-12-31 09:30:00 | 700.32 | 2025-01-06 13:15:00 | 672.29 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-02-24 09:30:00 | 575.69 | 2025-04-22 11:15:00 | 630.20 | STOP_HIT | 1.00 | -9.47% |
| SELL | retest2 | 2025-02-24 12:30:00 | 581.00 | 2025-04-22 11:15:00 | 630.20 | STOP_HIT | 1.00 | -8.47% |
| SELL | retest2 | 2025-02-28 09:15:00 | 580.83 | 2025-04-22 11:15:00 | 630.20 | STOP_HIT | 1.00 | -8.50% |
| SELL | retest2 | 2025-02-28 14:00:00 | 580.62 | 2025-04-22 11:15:00 | 630.20 | STOP_HIT | 1.00 | -8.54% |
| SELL | retest2 | 2025-05-14 14:00:00 | 610.95 | 2025-05-15 09:15:00 | 615.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-20 10:15:00 | 654.00 | 2025-08-05 09:15:00 | 719.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 12:30:00 | 654.70 | 2025-08-05 09:15:00 | 720.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 10:00:00 | 653.50 | 2025-08-05 09:15:00 | 718.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 10:30:00 | 655.00 | 2025-08-05 09:15:00 | 720.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 09:45:00 | 688.50 | 2025-08-05 09:15:00 | 757.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 10:15:00 | 686.75 | 2025-08-05 09:15:00 | 755.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 11:45:00 | 687.80 | 2025-08-05 09:15:00 | 756.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 13:00:00 | 686.75 | 2025-08-05 09:15:00 | 755.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 09:45:00 | 678.00 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-05 11:00:00 | 678.65 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-05 13:45:00 | 678.00 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-08 09:30:00 | 679.15 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-09 13:30:00 | 669.80 | 2025-09-18 11:15:00 | 736.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 09:15:00 | 675.05 | 2025-09-18 12:15:00 | 742.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 702.50 | 2026-02-24 09:15:00 | 667.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 702.50 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:45:00 | 705.40 | 2026-02-24 09:15:00 | 670.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:45:00 | 705.40 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2026-02-19 13:30:00 | 707.15 | 2026-02-24 09:15:00 | 671.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:30:00 | 707.15 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2026-02-20 15:15:00 | 705.80 | 2026-02-24 09:15:00 | 670.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 15:15:00 | 705.80 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.44% |
