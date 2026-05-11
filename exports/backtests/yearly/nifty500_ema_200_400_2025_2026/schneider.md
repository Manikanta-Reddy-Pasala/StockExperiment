# Schneider Electric Infrastructure Ltd. (SCHNEIDER)

## Backtest Summary

- **Window:** 2023-05-25 09:15:00 → 2026-05-08 15:30:00 (5067 bars)
- **Last close:** 1348.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 781.65 | 642.69 | 642.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 792.10 | 687.39 | 666.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 888.00 | 910.71 | 845.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:15:00 | 888.00 | 910.71 | 845.74 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 845.00 | 908.77 | 846.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 845.00 | 908.77 | 846.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 855.70 | 908.24 | 847.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:15:00 | 855.70 | 908.24 | 847.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 849.20 | 907.10 | 847.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:15:00 | 849.20 | 907.10 | 847.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 847.50 | 906.51 | 847.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:15:00 | 847.50 | 906.51 | 847.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 843.60 | 905.88 | 847.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:15:00 | 843.60 | 905.88 | 847.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 844.10 | 905.27 | 847.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:15:00 | 844.10 | 905.27 | 847.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 839.90 | 899.61 | 848.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:15:00 | 839.90 | 899.61 | 848.85 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 840.70 | 899.02 | 848.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:15:00 | 840.70 | 899.02 | 848.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 843.00 | 898.46 | 848.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:15:00 | 843.00 | 898.46 | 848.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 859.80 | 897.50 | 849.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 859.80 | 897.50 | 849.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 846.00 | 895.74 | 849.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:15:00 | 846.00 | 895.74 | 849.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 850.20 | 895.29 | 849.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:15:00 | 850.20 | 895.29 | 849.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 847.80 | 894.81 | 849.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:15:00 | 847.80 | 894.81 | 849.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 857.00 | 894.44 | 849.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 857.00 | 894.44 | 849.14 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 845.00 | 893.95 | 849.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:15:00 | 845.00 | 893.95 | 849.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 848.90 | 893.50 | 849.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:15:00 | 848.90 | 893.50 | 849.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 852.00 | 893.08 | 849.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:15:00 | 852.00 | 893.08 | 849.14 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 841.00 | 892.57 | 849.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:15:00 | 841.00 | 892.57 | 849.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 840.00 | 892.04 | 849.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:15:00 | 840.00 | 892.04 | 849.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 839.30 | 891.52 | 849.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:15:00 | 839.30 | 891.52 | 849.00 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 858.00 | 885.70 | 850.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:15:00 | 858.00 | 885.70 | 850.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 860.00 | 883.74 | 850.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 860.00 | 883.74 | 850.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 854.00 | 880.66 | 851.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 854.00 | 880.66 | 851.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 845.20 | 880.31 | 851.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:15:00 | 845.20 | 880.31 | 851.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 849.95 | 880.01 | 851.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:15:00 | 849.95 | 880.01 | 851.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 844.40 | 878.63 | 851.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 844.40 | 878.63 | 851.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 875.00 | 878.11 | 852.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:15:00 | 875.00 | 878.11 | 852.14 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 850.00 | 876.37 | 852.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:15:00 | 850.00 | 876.37 | 852.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 849.05 | 876.10 | 852.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 849.05 | 876.10 | 852.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 876.00 | 876.10 | 852.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 876.00 | 876.10 | 852.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 861.30 | 882.88 | 865.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 861.30 | 882.88 | 865.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 864.00 | 882.70 | 865.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:15:00 | 864.00 | 882.70 | 865.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 862.05 | 882.49 | 865.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:15:00 | 862.05 | 882.49 | 865.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 855.95 | 882.09 | 865.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:15:00 | 855.95 | 882.09 | 865.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 855.05 | 881.83 | 865.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:15:00 | 855.05 | 881.83 | 865.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 867.05 | 881.28 | 865.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 867.05 | 881.28 | 865.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 863.45 | 881.10 | 865.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:15:00 | 863.45 | 881.10 | 865.23 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 858.00 | 880.87 | 865.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:15:00 | 858.00 | 880.87 | 865.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 850.80 | 868.08 | 860.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:15:00 | 850.80 | 868.08 | 860.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 850.00 | 866.93 | 860.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:15:00 | 850.00 | 866.93 | 860.26 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 849.15 | 865.42 | 859.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:15:00 | 849.15 | 865.42 | 859.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 813.00 | 854.71 | 854.87 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 877.65 | 852.19 | 852.16 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 816.95 | 851.90 | 852.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 805.95 | 849.75 | 850.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 751.85 | 746.76 | 778.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 09:15:00 | 751.85 | 746.76 | 778.51 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 709.95 | 673.90 | 708.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 709.95 | 673.90 | 708.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 710.15 | 674.26 | 709.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:15:00 | 710.15 | 674.26 | 709.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 718.55 | 674.70 | 709.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:15:00 | 718.55 | 674.70 | 709.04 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 719.70 | 675.15 | 709.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:15:00 | 719.70 | 675.15 | 709.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 830.90 | 731.34 | 731.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 844.80 | 732.47 | 731.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 851.45 | 858.54 | 818.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 09:15:00 | 851.45 | 858.54 | 818.87 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |

