# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1088.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 43 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 36
- **Target hits / Stop hits / Partials:** 9 / 38 / 9
- **Avg / median % per leg:** 0.24% / -1.76%
- **Sum % (uncompounded):** 13.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 2 | 10.5% | 2 | 17 | 0 | -1.99% | -37.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 2 | 10.5% | 2 | 17 | 0 | -1.99% | -37.8% |
| SELL (all) | 37 | 18 | 48.6% | 7 | 21 | 9 | 1.39% | 51.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -9.34% | -37.4% |
| SELL @ 3rd Alert (retest2) | 33 | 18 | 54.5% | 7 | 17 | 9 | 2.69% | 88.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -9.34% | -37.4% |
| retest2 (combined) | 52 | 20 | 38.5% | 9 | 34 | 9 | 0.98% | 50.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 12:15:00 | 702.70 | 785.81 | 785.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 13:15:00 | 699.95 | 784.95 | 785.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 711.10 | 680.32 | 716.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 711.10 | 680.32 | 716.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 711.10 | 680.32 | 716.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:00:00 | 711.10 | 680.32 | 716.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 710.00 | 680.62 | 716.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:45:00 | 709.20 | 680.62 | 716.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 711.00 | 681.53 | 716.38 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 09:15:00 | 782.00 | 735.83 | 735.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 792.60 | 738.98 | 737.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 10:15:00 | 756.30 | 756.35 | 747.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:30:00 | 755.30 | 756.35 | 747.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 751.00 | 756.75 | 748.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 743.60 | 756.75 | 748.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 747.60 | 756.66 | 748.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 747.60 | 756.66 | 748.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 750.65 | 756.60 | 748.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:00:00 | 750.65 | 756.60 | 748.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 742.60 | 756.46 | 748.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:30:00 | 752.15 | 751.50 | 746.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:45:00 | 750.95 | 751.31 | 746.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 13:15:00 | 736.80 | 751.00 | 746.60 | SL hit (close<static) qty=1.00 sl=741.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 13:15:00 | 709.60 | 746.39 | 746.44 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 09:15:00 | 791.00 | 746.73 | 746.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 11:15:00 | 797.10 | 747.65 | 747.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 775.25 | 779.74 | 765.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 775.25 | 779.74 | 765.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 775.25 | 779.74 | 765.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 771.75 | 779.74 | 765.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 779.80 | 786.89 | 774.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 776.75 | 786.89 | 774.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 779.00 | 791.91 | 779.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 779.00 | 791.91 | 779.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 780.50 | 791.79 | 779.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 780.50 | 791.79 | 779.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 785.00 | 791.73 | 779.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:45:00 | 787.65 | 791.16 | 779.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 13:15:00 | 777.40 | 790.92 | 779.81 | SL hit (close<static) qty=1.00 sl=777.55 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 15:15:00 | 851.00 | 933.93 | 934.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 847.00 | 914.36 | 920.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 910.45 | 892.30 | 907.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 10:15:00 | 910.45 | 892.30 | 907.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 910.45 | 892.30 | 907.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 910.45 | 892.30 | 907.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 902.85 | 892.40 | 907.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 12:15:00 | 899.45 | 892.40 | 907.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 922.40 | 893.26 | 907.07 | SL hit (close>static) qty=1.00 sl=912.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 13:15:00 | 940.40 | 850.65 | 850.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 970.10 | 857.09 | 853.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 1050.70 | 1058.32 | 1012.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 1050.70 | 1058.32 | 1012.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1062.50 | 1081.85 | 1057.05 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 811.15 | 1039.94 | 1040.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 784.15 | 1037.39 | 1039.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 828.75 | 827.86 | 885.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:30:00 | 833.25 | 827.86 | 885.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 875.30 | 832.19 | 878.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:30:00 | 875.00 | 832.19 | 878.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 878.70 | 836.54 | 877.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 878.70 | 836.54 | 877.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 887.05 | 837.04 | 877.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 888.05 | 837.04 | 877.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 880.85 | 842.87 | 878.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:45:00 | 878.15 | 847.74 | 878.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 877.10 | 848.32 | 878.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 890.65 | 852.21 | 878.13 | SL hit (close>static) qty=1.00 sl=886.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 942.35 | 884.26 | 884.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 960.40 | 905.80 | 899.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 952.75 | 953.30 | 933.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 14:00:00 | 952.75 | 953.30 | 933.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 929.25 | 952.98 | 933.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 929.25 | 952.98 | 933.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 929.75 | 952.75 | 933.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 928.95 | 952.75 | 933.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 937.00 | 952.60 | 933.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:45:00 | 927.30 | 952.60 | 933.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 937.85 | 952.45 | 933.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:15:00 | 932.95 | 952.45 | 933.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 919.65 | 952.12 | 933.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 919.65 | 952.12 | 933.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 930.00 | 951.90 | 933.34 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 838.35 | 917.18 | 917.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 824.00 | 914.68 | 916.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 806.70 | 805.63 | 835.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 09:15:00 | 773.55 | 804.86 | 833.37 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 09:30:00 | 784.00 | 802.11 | 830.85 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 780.70 | 801.90 | 830.60 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 13:30:00 | 785.20 | 801.59 | 830.02 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 853.75 | 800.83 | 826.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 853.75 | 800.83 | 826.33 | SL hit (close>ema400) qty=1.00 sl=826.33 alert=retest1 |

### Cycle 10 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 979.70 | 844.90 | 844.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 987.85 | 847.65 | 845.65 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 12:30:00 | 752.15 | 2024-05-17 13:15:00 | 736.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-05-17 09:45:00 | 750.95 | 2024-05-17 13:15:00 | 736.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-05-23 09:15:00 | 766.40 | 2024-05-29 09:15:00 | 739.70 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-07-22 10:45:00 | 787.65 | 2024-07-22 13:15:00 | 777.40 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-07-22 14:30:00 | 786.80 | 2024-07-23 12:15:00 | 757.10 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-07-25 14:45:00 | 787.25 | 2024-08-05 10:15:00 | 774.05 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-07-26 09:15:00 | 792.60 | 2024-08-05 10:15:00 | 774.05 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-31 09:45:00 | 790.35 | 2024-08-05 10:15:00 | 774.05 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-07-31 10:30:00 | 789.75 | 2024-08-05 10:15:00 | 774.05 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-08-06 09:15:00 | 796.40 | 2024-08-21 09:15:00 | 876.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 14:45:00 | 791.95 | 2024-08-21 09:15:00 | 871.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-08 11:00:00 | 948.60 | 2024-10-18 09:15:00 | 908.15 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2024-10-09 09:15:00 | 963.85 | 2024-10-18 09:15:00 | 908.15 | STOP_HIT | 1.00 | -5.78% |
| BUY | retest2 | 2024-10-17 10:30:00 | 947.90 | 2024-10-18 09:15:00 | 908.15 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-10-23 12:45:00 | 949.50 | 2024-10-24 14:15:00 | 932.65 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-10-24 10:15:00 | 983.30 | 2024-10-28 10:15:00 | 932.15 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2024-10-25 09:15:00 | 988.25 | 2024-10-28 10:15:00 | 932.15 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2024-10-25 10:45:00 | 969.45 | 2024-10-28 12:15:00 | 920.95 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2024-10-31 15:00:00 | 973.90 | 2024-11-13 09:15:00 | 921.45 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2025-01-01 12:15:00 | 899.45 | 2025-01-02 09:15:00 | 922.40 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-01-06 13:45:00 | 897.15 | 2025-01-10 09:15:00 | 852.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 15:15:00 | 886.00 | 2025-01-10 09:15:00 | 853.53 | PARTIAL | 0.50 | 3.67% |
| SELL | retest2 | 2025-01-07 12:15:00 | 898.45 | 2025-01-13 09:15:00 | 841.70 | PARTIAL | 0.50 | 6.32% |
| SELL | retest2 | 2025-01-06 13:45:00 | 897.15 | 2025-01-13 11:15:00 | 807.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 15:15:00 | 886.00 | 2025-01-13 11:15:00 | 808.61 | TARGET_HIT | 0.50 | 8.74% |
| SELL | retest2 | 2025-01-07 12:15:00 | 898.45 | 2025-01-13 14:15:00 | 797.40 | TARGET_HIT | 0.50 | 11.25% |
| SELL | retest2 | 2025-01-17 09:15:00 | 890.85 | 2025-01-17 12:15:00 | 910.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-01-17 10:00:00 | 891.05 | 2025-01-17 12:15:00 | 910.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-01-22 09:15:00 | 888.00 | 2025-01-22 14:15:00 | 917.25 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-01-23 09:15:00 | 891.35 | 2025-01-27 09:15:00 | 846.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 888.90 | 2025-01-27 09:15:00 | 844.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 886.30 | 2025-01-27 09:15:00 | 841.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 891.35 | 2025-01-27 10:15:00 | 802.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 888.90 | 2025-01-27 10:15:00 | 800.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 886.30 | 2025-01-27 10:15:00 | 797.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 09:30:00 | 886.75 | 2025-02-01 10:15:00 | 911.15 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-02-01 13:00:00 | 880.00 | 2025-02-05 11:15:00 | 895.50 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-02-03 09:15:00 | 882.25 | 2025-02-05 11:15:00 | 895.50 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-02-03 10:15:00 | 883.45 | 2025-02-06 14:15:00 | 907.35 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-02-10 09:15:00 | 884.85 | 2025-02-11 09:15:00 | 840.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 884.85 | 2025-02-12 09:15:00 | 796.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 10:15:00 | 883.00 | 2025-04-02 10:15:00 | 901.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-09-26 13:45:00 | 878.15 | 2025-10-01 13:15:00 | 890.65 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-29 09:30:00 | 877.10 | 2025-10-01 13:15:00 | 890.65 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-10-08 11:00:00 | 878.10 | 2025-10-17 09:15:00 | 834.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 12:30:00 | 878.00 | 2025-10-17 09:15:00 | 834.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 11:00:00 | 878.10 | 2025-10-20 10:15:00 | 858.20 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2025-10-08 12:30:00 | 878.00 | 2025-10-20 10:15:00 | 858.20 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2025-10-08 15:00:00 | 874.70 | 2025-10-23 09:15:00 | 889.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-10-09 10:00:00 | 874.00 | 2025-10-23 09:15:00 | 889.40 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-09 12:15:00 | 875.05 | 2025-10-23 09:15:00 | 889.40 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-10 09:15:00 | 873.25 | 2025-10-23 09:15:00 | 889.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest1 | 2026-03-30 09:15:00 | 773.55 | 2026-04-08 09:15:00 | 853.75 | STOP_HIT | 1.00 | -10.37% |
| SELL | retest1 | 2026-04-01 09:30:00 | 784.00 | 2026-04-08 09:15:00 | 853.75 | STOP_HIT | 1.00 | -8.90% |
| SELL | retest1 | 2026-04-01 11:00:00 | 780.70 | 2026-04-08 09:15:00 | 853.75 | STOP_HIT | 1.00 | -9.36% |
| SELL | retest1 | 2026-04-01 13:30:00 | 785.20 | 2026-04-08 09:15:00 | 853.75 | STOP_HIT | 1.00 | -8.73% |
