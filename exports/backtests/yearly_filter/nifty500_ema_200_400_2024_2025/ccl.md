# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1122.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 42 |
| PARTIAL | 1 |
| TARGET_HIT | 9 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 36
- **Target hits / Stop hits / Partials:** 9 / 36 / 1
- **Avg / median % per leg:** -0.63% / -2.34%
- **Sum % (uncompounded):** -28.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 8 | 26.7% | 8 | 22 | 0 | 0.45% | 13.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.46% | -16.4% |
| BUY @ 3rd Alert (retest2) | 27 | 8 | 29.6% | 8 | 19 | 0 | 1.11% | 30.0% |
| SELL (all) | 16 | 2 | 12.5% | 1 | 14 | 1 | -2.65% | -42.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 1 | 14 | 1 | -2.65% | -42.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.46% | -16.4% |
| retest2 (combined) | 43 | 10 | 23.3% | 9 | 33 | 1 | -0.29% | -12.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 601.70 | 591.56 | 591.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 11:15:00 | 608.15 | 592.80 | 592.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 593.55 | 594.85 | 593.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 593.55 | 594.85 | 593.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 593.55 | 594.85 | 593.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:45:00 | 593.20 | 594.85 | 593.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 596.00 | 594.86 | 593.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 597.95 | 594.89 | 593.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 600.00 | 594.89 | 593.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 589.30 | 595.57 | 593.92 | SL hit (close<static) qty=1.00 sl=592.60 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 628.45 | 679.13 | 679.26 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 708.75 | 675.93 | 675.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 741.15 | 684.53 | 680.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 751.30 | 754.89 | 730.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 751.30 | 754.89 | 730.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 725.60 | 752.89 | 731.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 725.60 | 752.89 | 731.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 725.00 | 752.61 | 731.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 724.25 | 752.61 | 731.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 732.15 | 751.28 | 731.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 732.90 | 751.28 | 731.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 730.35 | 751.07 | 731.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 730.35 | 751.07 | 731.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 729.40 | 750.85 | 731.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 730.95 | 750.85 | 731.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 733.60 | 750.50 | 731.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 735.00 | 750.33 | 731.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:30:00 | 739.45 | 750.11 | 731.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 726.20 | 749.27 | 731.49 | SL hit (close<static) qty=1.00 sl=729.60 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 635.25 | 720.36 | 720.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 627.00 | 703.87 | 711.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 659.05 | 657.69 | 680.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 659.05 | 657.69 | 680.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 677.25 | 657.89 | 680.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 679.65 | 657.89 | 680.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 675.15 | 658.26 | 680.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 656.00 | 660.45 | 680.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:15:00 | 623.20 | 659.28 | 678.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 590.40 | 657.31 | 676.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 773.00 | 612.53 | 611.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 13:15:00 | 819.25 | 673.45 | 646.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 803.70 | 803.76 | 747.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 803.05 | 803.76 | 747.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 849.25 | 863.10 | 830.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 870.00 | 861.96 | 832.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:30:00 | 870.00 | 862.04 | 833.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 871.20 | 861.22 | 836.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 889.75 | 861.52 | 836.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-03 10:15:00 | 957.00 | 877.50 | 854.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 837.85 | 865.64 | 865.68 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 988.15 | 864.26 | 863.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1012.45 | 870.19 | 866.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 970.40 | 971.04 | 934.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:45:00 | 988.70 | 971.44 | 935.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 992.80 | 971.84 | 936.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:15:00 | 990.90 | 973.71 | 938.77 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | SL hit (close<ema400) qty=1.00 sl=939.94 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-29 10:00:00 | 585.05 | 2024-05-29 11:15:00 | 592.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-29 15:00:00 | 585.95 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2024-05-30 11:00:00 | 585.70 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2024-05-30 14:15:00 | 584.85 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-06-03 10:30:00 | 574.65 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.67% |
| SELL | retest2 | 2024-06-03 12:45:00 | 575.30 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2024-06-03 13:30:00 | 575.05 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.60% |
| SELL | retest2 | 2024-06-03 14:00:00 | 574.90 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.63% |
| SELL | retest2 | 2024-06-07 15:15:00 | 598.00 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-06-10 11:15:00 | 602.25 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-06-11 12:30:00 | 602.10 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-06-11 13:30:00 | 601.40 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-18 10:30:00 | 597.95 | 2024-07-19 14:15:00 | 589.30 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-07-18 11:15:00 | 600.00 | 2024-07-19 14:15:00 | 589.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-07-23 10:15:00 | 596.90 | 2024-07-31 12:15:00 | 656.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 12:30:00 | 598.90 | 2024-07-31 13:15:00 | 658.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 15:15:00 | 735.00 | 2024-12-31 09:15:00 | 726.20 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-12-30 09:30:00 | 739.45 | 2024-12-31 09:15:00 | 726.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-31 14:30:00 | 738.95 | 2025-01-01 15:15:00 | 727.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-01-01 10:00:00 | 734.55 | 2025-01-01 15:15:00 | 727.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-02-06 09:15:00 | 656.00 | 2025-02-10 10:15:00 | 623.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 656.00 | 2025-02-11 09:15:00 | 590.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 673.85 | 2025-05-06 14:15:00 | 694.25 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-05-06 13:00:00 | 673.70 | 2025-05-06 14:15:00 | 694.25 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-08-08 15:15:00 | 870.00 | 2025-09-03 10:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 14:30:00 | 870.00 | 2025-09-03 10:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 871.20 | 2025-09-03 10:15:00 | 958.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:45:00 | 889.75 | 2025-10-13 10:15:00 | 820.60 | STOP_HIT | 1.00 | -7.77% |
| BUY | retest1 | 2025-12-02 14:45:00 | 988.70 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest1 | 2025-12-03 09:30:00 | 992.80 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2025-12-04 11:15:00 | 990.90 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2025-12-05 14:45:00 | 951.10 | 2025-12-08 12:15:00 | 932.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-08 09:30:00 | 965.20 | 2025-12-08 12:15:00 | 932.80 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-12-09 10:15:00 | 950.50 | 2025-12-29 10:15:00 | 929.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-12-23 14:15:00 | 950.30 | 2025-12-29 10:15:00 | 929.20 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-01-13 15:15:00 | 975.00 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-14 10:30:00 | 972.75 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-01-16 09:30:00 | 973.20 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-16 14:45:00 | 972.40 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-01-21 15:15:00 | 953.00 | 2026-01-22 15:15:00 | 930.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-01-27 14:45:00 | 951.10 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-28 10:30:00 | 952.10 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-01-28 11:00:00 | 950.80 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-02-02 14:00:00 | 972.15 | 2026-02-23 13:15:00 | 1069.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 14:45:00 | 966.75 | 2026-02-23 13:15:00 | 1063.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 11:30:00 | 971.80 | 2026-02-23 13:15:00 | 1068.98 | TARGET_HIT | 1.00 | 10.00% |
