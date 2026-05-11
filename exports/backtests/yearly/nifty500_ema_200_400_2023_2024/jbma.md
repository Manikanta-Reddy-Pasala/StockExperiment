# JBM Auto Ltd. (JBMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 649.00
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
| ALERT2 | 10 |
| ALERT2_SKIP | 7 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 15 |
| TARGET_HIT | 17 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 25
- **Target hits / Stop hits / Partials:** 17 / 26 / 15
- **Avg / median % per leg:** 2.68% / 5.00%
- **Sum % (uncompounded):** 155.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.50% | 18.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.50% | 18.0% |
| SELL (all) | 46 | 29 | 63.0% | 13 | 18 | 15 | 2.99% | 137.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 29 | 63.0% | 13 | 18 | 15 | 2.99% | 137.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 33 | 56.9% | 17 | 26 | 15 | 2.68% | 155.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 613.50 | 668.12 | 668.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 603.95 | 663.66 | 665.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 10:15:00 | 622.00 | 619.04 | 635.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 645.00 | 619.81 | 635.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 645.00 | 619.81 | 635.00 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 15:15:00 | 695.50 | 640.96 | 640.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 709.13 | 642.77 | 641.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 1028.50 | 1033.24 | 949.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 951.00 | 1028.38 | 951.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 951.00 | 1028.38 | 951.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 889.10 | 932.08 | 929.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 861.45 | 926.37 | 926.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 854.58 | 925.01 | 925.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 11:15:00 | 916.00 | 910.87 | 917.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 11:15:00 | 916.00 | 910.87 | 917.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 916.00 | 910.87 | 917.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 12:00:00 | 916.00 | 910.87 | 917.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 914.98 | 911.03 | 917.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 10:45:00 | 909.50 | 911.00 | 917.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 14:30:00 | 910.50 | 911.03 | 917.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 894.75 | 911.04 | 917.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 933.63 | 910.05 | 916.86 | SL hit (close>static) qty=1.00 sl=920.45 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 14:15:00 | 985.50 | 923.09 | 922.91 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 907.70 | 923.32 | 923.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 10:15:00 | 904.00 | 922.02 | 922.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 935.60 | 921.31 | 922.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 935.60 | 921.31 | 922.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 935.60 | 921.31 | 922.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 935.60 | 921.31 | 922.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 934.95 | 921.44 | 922.32 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 967.10 | 923.31 | 923.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 973.40 | 923.81 | 923.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 956.28 | 969.38 | 949.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 956.28 | 969.38 | 949.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 956.28 | 969.38 | 949.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 956.28 | 969.38 | 949.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 940.00 | 969.08 | 949.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 940.00 | 969.08 | 949.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 983.48 | 969.23 | 949.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:45:00 | 1008.03 | 969.48 | 949.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 1033.18 | 970.24 | 950.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-07 12:15:00 | 1108.83 | 981.57 | 957.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 925.58 | 1006.88 | 1007.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 909.05 | 1001.26 | 1004.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 14:15:00 | 976.83 | 976.47 | 988.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 14:15:00 | 976.83 | 976.47 | 988.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 976.83 | 976.47 | 988.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:45:00 | 990.43 | 976.47 | 988.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 974.85 | 971.02 | 983.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:00:00 | 956.05 | 971.09 | 982.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 960.13 | 968.08 | 980.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 956.73 | 967.97 | 980.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 958.50 | 967.58 | 979.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1002.55 | 967.84 | 979.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 1002.55 | 967.84 | 979.61 | SL hit (close>static) qty=1.00 sl=994.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 686.00 | 647.60 | 647.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 707.35 | 655.63 | 651.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 704.60 | 707.32 | 688.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 704.60 | 707.32 | 688.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 705.95 | 707.39 | 688.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:45:00 | 706.25 | 707.34 | 688.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 724.80 | 706.48 | 689.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 707.00 | 706.61 | 690.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 677.55 | 705.29 | 690.45 | SL hit (close<static) qty=1.00 sl=688.90 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 643.20 | 679.68 | 679.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 640.00 | 673.70 | 676.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 658.30 | 654.93 | 663.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 658.30 | 654.93 | 663.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 670.15 | 655.08 | 663.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 672.70 | 655.08 | 663.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 671.60 | 655.25 | 663.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 668.40 | 655.25 | 663.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 660.50 | 655.58 | 664.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 660.00 | 655.58 | 664.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 13:00:00 | 660.15 | 655.65 | 663.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 627.00 | 653.51 | 661.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 627.14 | 653.51 | 661.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-08 14:15:00 | 594.00 | 641.66 | 653.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 754.35 | 644.98 | 644.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 765.85 | 666.12 | 656.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 14:15:00 | 678.10 | 678.33 | 664.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 678.10 | 678.33 | 664.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 666.35 | 677.59 | 666.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 664.75 | 677.59 | 666.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 661.80 | 677.43 | 666.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 661.80 | 677.43 | 666.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 662.70 | 677.28 | 666.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 661.20 | 677.28 | 666.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 661.50 | 674.83 | 665.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 661.65 | 674.83 | 665.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 663.00 | 674.48 | 665.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 665.00 | 674.48 | 665.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 653.85 | 674.04 | 665.67 | SL hit (close<static) qty=1.00 sl=661.05 alert=retest2 |

### Cycle 11 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 632.00 | 660.78 | 660.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 630.60 | 659.97 | 660.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 625.95 | 593.02 | 614.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 631.30 | 593.02 | 614.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 642.05 | 593.51 | 614.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 642.05 | 593.51 | 614.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 641.00 | 598.82 | 615.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 644.05 | 598.82 | 615.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 643.60 | 599.27 | 615.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 643.60 | 599.27 | 615.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 631.60 | 618.75 | 622.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 635.35 | 618.75 | 622.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 606.95 | 617.25 | 621.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 601.00 | 616.74 | 621.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 601.00 | 616.28 | 620.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 601.20 | 616.28 | 620.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 602.25 | 616.14 | 620.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 572.14 | 611.05 | 617.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 570.95 | 610.62 | 617.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 570.95 | 610.62 | 617.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 571.14 | 610.62 | 617.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 10:15:00 | 540.90 | 601.36 | 611.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 12:15:00 | 619.60 | 570.37 | 570.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 628.25 | 572.52 | 571.39 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-25 10:45:00 | 909.50 | 2024-04-29 09:15:00 | 933.63 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-04-25 14:30:00 | 910.50 | 2024-04-29 09:15:00 | 933.63 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-04-26 09:15:00 | 894.75 | 2024-04-29 09:15:00 | 933.63 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2024-06-04 13:45:00 | 1008.03 | 2024-06-07 12:15:00 | 1108.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1033.18 | 2024-07-01 13:15:00 | 1136.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 15:15:00 | 1047.50 | 2024-07-01 13:15:00 | 1109.77 | TARGET_HIT | 1.00 | 5.94% |
| BUY | retest2 | 2024-06-20 13:30:00 | 1008.88 | 2024-07-08 12:15:00 | 1152.25 | TARGET_HIT | 1.00 | 14.21% |
| BUY | retest2 | 2024-07-26 09:45:00 | 1022.00 | 2024-08-02 09:15:00 | 995.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-09-06 10:00:00 | 956.05 | 2024-09-12 09:15:00 | 1002.55 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2024-09-11 09:15:00 | 960.13 | 2024-09-12 09:15:00 | 1002.55 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2024-09-11 10:00:00 | 956.73 | 2024-09-12 09:15:00 | 1002.55 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2024-09-11 15:15:00 | 958.50 | 2024-09-12 09:15:00 | 1002.55 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2024-09-19 11:00:00 | 972.48 | 2024-09-24 11:15:00 | 923.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 14:30:00 | 972.20 | 2024-09-24 11:15:00 | 923.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 10:30:00 | 967.43 | 2024-09-25 09:15:00 | 919.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 11:00:00 | 972.48 | 2024-10-07 09:15:00 | 875.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-19 14:30:00 | 972.20 | 2024-10-07 09:15:00 | 874.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-20 10:30:00 | 967.43 | 2024-10-07 10:15:00 | 870.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-13 10:45:00 | 706.25 | 2025-06-19 12:15:00 | 677.55 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-06-16 14:45:00 | 724.80 | 2025-06-19 12:15:00 | 677.55 | STOP_HIT | 1.00 | -6.52% |
| BUY | retest2 | 2025-06-18 09:15:00 | 707.00 | 2025-06-19 12:15:00 | 677.55 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-07-25 11:15:00 | 660.00 | 2025-07-31 09:15:00 | 627.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 13:00:00 | 660.15 | 2025-07-31 09:15:00 | 627.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 11:15:00 | 660.00 | 2025-08-08 14:15:00 | 594.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-25 13:00:00 | 660.15 | 2025-08-08 14:15:00 | 594.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-25 12:30:00 | 650.95 | 2025-08-26 14:15:00 | 618.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:30:00 | 650.95 | 2025-09-08 11:15:00 | 626.70 | STOP_HIT | 0.50 | 3.73% |
| BUY | retest2 | 2025-10-10 14:15:00 | 665.00 | 2025-10-13 09:15:00 | 653.85 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-29 09:15:00 | 666.55 | 2025-11-03 11:15:00 | 658.70 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-03 09:15:00 | 664.30 | 2025-11-03 11:15:00 | 658.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-03 09:45:00 | 665.90 | 2025-11-03 11:15:00 | 658.70 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-13 14:00:00 | 601.00 | 2026-01-20 12:15:00 | 572.14 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2026-01-14 10:30:00 | 601.00 | 2026-01-20 13:15:00 | 570.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 601.20 | 2026-01-20 13:15:00 | 570.95 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-01-14 12:00:00 | 602.25 | 2026-01-20 13:15:00 | 571.14 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-01-13 14:00:00 | 601.00 | 2026-01-23 10:15:00 | 540.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 601.00 | 2026-01-23 10:15:00 | 540.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 601.20 | 2026-01-23 10:15:00 | 541.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 12:00:00 | 602.25 | 2026-01-23 10:15:00 | 542.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 561.85 | 2026-03-02 09:15:00 | 533.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 10:00:00 | 566.30 | 2026-03-02 09:15:00 | 537.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:00:00 | 567.05 | 2026-03-02 09:15:00 | 538.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:30:00 | 566.40 | 2026-03-02 09:15:00 | 538.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 561.85 | 2026-03-04 09:15:00 | 509.67 | TARGET_HIT | 0.50 | 9.29% |
| SELL | retest2 | 2026-02-20 10:00:00 | 566.30 | 2026-03-04 09:15:00 | 510.34 | TARGET_HIT | 0.50 | 9.88% |
| SELL | retest2 | 2026-02-20 11:00:00 | 567.05 | 2026-03-04 09:15:00 | 509.76 | TARGET_HIT | 0.50 | 10.10% |
| SELL | retest2 | 2026-02-20 11:30:00 | 566.40 | 2026-03-04 12:15:00 | 505.67 | TARGET_HIT | 0.50 | 10.72% |
| SELL | retest2 | 2026-03-19 10:15:00 | 561.45 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-03-19 11:00:00 | 561.65 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2026-03-19 11:45:00 | 559.95 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2026-03-20 09:30:00 | 554.45 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-03-20 12:15:00 | 551.30 | 2026-03-20 13:15:00 | 580.70 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2026-03-23 09:15:00 | 548.40 | 2026-03-24 09:15:00 | 575.05 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2026-03-27 10:00:00 | 551.20 | 2026-03-30 13:15:00 | 523.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 10:00:00 | 551.20 | 2026-04-01 09:15:00 | 573.55 | STOP_HIT | 0.50 | -4.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 543.60 | 2026-04-02 13:15:00 | 567.15 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2026-04-06 09:15:00 | 552.55 | 2026-04-08 09:15:00 | 579.50 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-04-06 11:00:00 | 556.40 | 2026-04-08 09:15:00 | 579.50 | STOP_HIT | 1.00 | -4.15% |
