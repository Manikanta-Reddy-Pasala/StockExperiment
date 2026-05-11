# Jindal Stainless Ltd. (JSL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 753.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 38 |
| PARTIAL | 3 |
| TARGET_HIT | 7 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 31
- **Target hits / Stop hits / Partials:** 7 / 33 / 3
- **Avg / median % per leg:** 0.34% / -1.73%
- **Sum % (uncompounded):** 14.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 6 | 8 | 0 | 2.53% | 35.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 6 | 8 | 0 | 2.53% | 35.4% |
| SELL (all) | 29 | 6 | 20.7% | 1 | 25 | 3 | -0.71% | -20.6% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.07% | 12.3% |
| SELL @ 3rd Alert (retest2) | 25 | 2 | 8.0% | 1 | 23 | 1 | -1.32% | -32.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.07% | 12.3% |
| retest2 (combined) | 39 | 8 | 20.5% | 7 | 31 | 1 | 0.06% | 2.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 13:15:00 | 680.80 | 752.62 | 752.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 669.35 | 750.33 | 751.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 729.05 | 726.69 | 737.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 729.05 | 726.69 | 737.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 737.70 | 726.93 | 737.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 736.70 | 726.93 | 737.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 742.55 | 727.08 | 737.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 742.55 | 727.08 | 737.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 740.15 | 727.21 | 737.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 740.15 | 727.21 | 737.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 745.55 | 727.68 | 738.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 745.55 | 727.68 | 738.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 730.20 | 728.78 | 737.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:45:00 | 729.55 | 728.81 | 737.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 15:15:00 | 727.00 | 729.08 | 737.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 09:45:00 | 729.20 | 729.05 | 737.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 14:15:00 | 744.00 | 729.34 | 737.69 | SL hit (close>static) qty=1.00 sl=739.80 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 766.20 | 740.16 | 740.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 779.00 | 740.55 | 740.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 756.80 | 757.66 | 750.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 14:30:00 | 754.95 | 757.66 | 750.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 750.00 | 757.56 | 750.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 11:00:00 | 757.80 | 757.56 | 750.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 12:30:00 | 758.65 | 757.61 | 750.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:30:00 | 757.65 | 757.50 | 750.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:45:00 | 759.10 | 756.57 | 750.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 754.35 | 756.91 | 751.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 758.85 | 756.91 | 751.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:00:00 | 760.45 | 756.95 | 751.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 758.65 | 756.94 | 751.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:30:00 | 759.70 | 756.77 | 751.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 767.95 | 757.53 | 752.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 735.10 | 757.53 | 752.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 745.75 | 757.42 | 752.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 745.75 | 757.42 | 752.09 | SL hit (close<static) qty=1.00 sl=749.55 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 652.00 | 746.72 | 747.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 645.65 | 734.66 | 740.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 712.85 | 712.06 | 726.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:30:00 | 706.55 | 712.13 | 725.99 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:00:00 | 708.35 | 712.13 | 725.99 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 724.10 | 712.53 | 725.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 724.10 | 712.53 | 725.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 671.22 | 706.05 | 719.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 672.93 | 706.05 | 719.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-28 13:15:00 | 699.35 | 697.44 | 712.23 | SL hit (close>ema200) qty=0.50 sl=697.44 alert=retest1 |

### Cycle 4 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 752.00 | 719.84 | 719.72 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 690.55 | 720.54 | 720.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 10:15:00 | 685.30 | 720.19 | 720.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 12:15:00 | 630.70 | 623.90 | 648.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 12:45:00 | 630.75 | 623.90 | 648.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 642.85 | 614.24 | 637.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 10:00:00 | 642.85 | 614.24 | 637.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 643.90 | 614.54 | 637.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:00:00 | 635.50 | 622.98 | 638.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 637.00 | 623.39 | 638.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 633.50 | 623.86 | 638.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:45:00 | 636.65 | 624.11 | 638.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 638.65 | 624.25 | 638.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 638.65 | 624.25 | 638.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 639.20 | 624.40 | 638.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 639.00 | 624.40 | 638.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 641.20 | 624.57 | 638.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 641.20 | 624.57 | 638.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 643.90 | 624.76 | 638.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:45:00 | 643.65 | 624.76 | 638.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 644.25 | 625.45 | 638.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 647.65 | 625.45 | 638.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 648.00 | 625.68 | 638.53 | SL hit (close>static) qty=1.00 sl=645.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 646.45 | 606.49 | 606.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 660.70 | 613.28 | 609.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 663.50 | 667.10 | 645.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:00:00 | 663.50 | 667.10 | 645.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 669.70 | 682.05 | 668.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 669.70 | 682.05 | 668.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 666.00 | 681.89 | 668.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 666.00 | 681.89 | 668.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 665.00 | 681.72 | 668.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 664.75 | 681.72 | 668.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 658.60 | 678.74 | 668.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 658.60 | 678.74 | 668.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 675.00 | 678.50 | 668.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 678.20 | 678.48 | 668.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:15:00 | 684.15 | 690.95 | 677.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:30:00 | 678.90 | 690.43 | 677.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 678.75 | 690.43 | 677.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 676.85 | 690.30 | 677.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 682.95 | 690.30 | 677.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 687.85 | 690.27 | 678.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 690.30 | 690.27 | 678.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:45:00 | 690.00 | 690.29 | 678.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 10:15:00 | 746.02 | 696.50 | 682.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 759.05 | 784.17 | 784.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 751.85 | 783.59 | 783.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 782.20 | 779.59 | 781.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 11:15:00 | 782.20 | 779.59 | 781.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 782.20 | 779.59 | 781.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 782.20 | 779.59 | 781.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 792.05 | 779.72 | 781.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 790.70 | 779.72 | 781.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 789.35 | 779.81 | 781.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:15:00 | 792.75 | 779.81 | 781.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 809.90 | 783.80 | 783.78 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 777.40 | 783.74 | 783.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 773.80 | 783.62 | 783.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 785.55 | 783.62 | 783.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 15:15:00 | 785.55 | 783.62 | 783.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 785.55 | 783.62 | 783.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 758.00 | 783.62 | 783.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 720.10 | 778.21 | 780.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-16 10:15:00 | 682.20 | 762.19 | 771.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 771.10 | 760.62 | 760.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 783.25 | 761.68 | 761.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 763.10 | 764.37 | 762.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 763.10 | 764.37 | 762.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 763.10 | 764.37 | 762.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 756.80 | 764.37 | 762.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 771.95 | 764.45 | 762.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 763.30 | 764.45 | 762.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 761.25 | 765.01 | 762.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 761.25 | 765.01 | 762.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 764.70 | 765.01 | 762.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 761.55 | 765.01 | 762.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 761.70 | 764.98 | 762.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 761.70 | 764.98 | 762.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 762.30 | 764.95 | 762.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:30:00 | 762.30 | 764.95 | 762.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 763.45 | 764.93 | 762.98 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-23 10:45:00 | 729.55 | 2024-08-26 14:15:00 | 744.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-08-23 15:15:00 | 727.00 | 2024-08-26 14:15:00 | 744.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-08-26 09:45:00 | 729.20 | 2024-08-26 14:15:00 | 744.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-08-28 09:45:00 | 729.55 | 2024-08-29 15:15:00 | 740.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-08-30 09:15:00 | 730.90 | 2024-08-30 10:15:00 | 744.15 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-09-03 10:15:00 | 733.90 | 2024-09-10 13:15:00 | 743.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-09-03 12:45:00 | 733.40 | 2024-09-10 13:15:00 | 743.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-09-03 13:30:00 | 733.40 | 2024-09-10 13:15:00 | 743.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-09-09 09:15:00 | 720.85 | 2024-09-10 13:15:00 | 743.70 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-09-09 11:00:00 | 721.10 | 2024-09-10 13:15:00 | 743.70 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-09-09 13:00:00 | 721.40 | 2024-09-10 13:15:00 | 743.70 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2024-10-08 11:00:00 | 757.80 | 2024-10-18 09:15:00 | 745.75 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-10-08 12:30:00 | 758.65 | 2024-10-18 09:15:00 | 745.75 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-10-09 09:30:00 | 757.65 | 2024-10-18 09:15:00 | 745.75 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-10-10 13:45:00 | 759.10 | 2024-10-18 09:15:00 | 745.75 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-10-14 09:15:00 | 758.85 | 2024-10-21 09:15:00 | 725.25 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2024-10-14 10:00:00 | 760.45 | 2024-10-21 09:15:00 | 725.25 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2024-10-14 11:15:00 | 758.65 | 2024-10-21 09:15:00 | 725.25 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-10-15 13:30:00 | 759.70 | 2024-10-21 09:15:00 | 725.25 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest1 | 2024-11-07 10:30:00 | 706.55 | 2024-11-21 09:15:00 | 671.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-07 11:00:00 | 708.35 | 2024-11-21 09:15:00 | 672.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-07 10:30:00 | 706.55 | 2024-11-28 13:15:00 | 699.35 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-11-07 11:00:00 | 708.35 | 2024-11-28 13:15:00 | 699.35 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-03-12 10:00:00 | 635.50 | 2025-03-18 11:15:00 | 648.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-03-12 15:15:00 | 637.00 | 2025-03-18 11:15:00 | 648.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-03-13 14:15:00 | 633.50 | 2025-03-18 11:15:00 | 648.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-03-17 09:45:00 | 636.65 | 2025-03-18 11:15:00 | 648.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-08 14:45:00 | 589.30 | 2025-05-12 09:15:00 | 605.30 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-05-09 09:45:00 | 591.35 | 2025-05-12 09:15:00 | 605.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-05-09 10:30:00 | 591.70 | 2025-05-12 09:15:00 | 605.30 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-05-09 11:15:00 | 592.80 | 2025-05-12 09:15:00 | 605.30 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-12 11:30:00 | 596.10 | 2025-05-12 12:15:00 | 609.80 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-07-31 11:15:00 | 678.20 | 2025-08-18 10:15:00 | 746.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 10:15:00 | 684.15 | 2025-08-18 14:15:00 | 746.79 | TARGET_HIT | 1.00 | 9.16% |
| BUY | retest2 | 2025-08-11 14:30:00 | 678.90 | 2025-08-18 14:15:00 | 746.63 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2025-08-11 15:00:00 | 678.75 | 2025-08-19 09:15:00 | 752.57 | TARGET_HIT | 1.00 | 10.88% |
| BUY | retest2 | 2025-08-12 10:15:00 | 690.30 | 2025-08-19 09:15:00 | 759.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 10:45:00 | 690.00 | 2025-08-19 09:15:00 | 759.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 758.00 | 2026-03-09 09:15:00 | 720.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 758.00 | 2026-03-16 10:15:00 | 682.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 768.35 | 2026-04-16 14:15:00 | 789.65 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-13 12:00:00 | 780.30 | 2026-04-16 14:15:00 | 789.65 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-15 15:15:00 | 779.80 | 2026-04-16 14:15:00 | 789.65 | STOP_HIT | 1.00 | -1.26% |
