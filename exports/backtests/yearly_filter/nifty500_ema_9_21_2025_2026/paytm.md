# One 97 Communications Ltd. (PAYTM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 63 |
| ALERT1 | 46 |
| ALERT2 | 47 |
| ALERT2_SKIP | 23 |
| ALERT3 | 121 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 64 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 45
- **Target hits / Stop hits / Partials:** 2 / 62 / 10
- **Avg / median % per leg:** 0.40% / -0.87%
- **Sum % (uncompounded):** 29.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 9 | 27.3% | 1 | 32 | 0 | -0.68% | -22.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 9 | 27.3% | 1 | 32 | 0 | -0.68% | -22.3% |
| SELL (all) | 41 | 20 | 48.8% | 1 | 30 | 10 | 1.26% | 51.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 20 | 48.8% | 1 | 30 | 10 | 1.26% | 51.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 74 | 29 | 39.2% | 2 | 62 | 10 | 0.40% | 29.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 857.85 | 847.88 | 846.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 864.30 | 851.16 | 848.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 850.95 | 854.79 | 851.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 850.95 | 854.79 | 851.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 850.95 | 854.79 | 851.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 11:45:00 | 853.80 | 853.43 | 851.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 853.30 | 853.43 | 851.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:45:00 | 859.50 | 853.36 | 851.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 09:15:00 | 836.45 | 850.54 | 850.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 09:15:00 | 836.45 | 850.54 | 850.62 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 858.20 | 850.94 | 850.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 862.25 | 854.96 | 853.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 860.35 | 866.46 | 861.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 12:15:00 | 860.35 | 866.46 | 861.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 860.35 | 866.46 | 861.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 860.35 | 866.46 | 861.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 862.30 | 865.63 | 861.51 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 851.65 | 858.98 | 859.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 828.05 | 842.23 | 849.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 832.50 | 832.31 | 839.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 832.50 | 832.31 | 839.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 843.25 | 834.50 | 839.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 843.25 | 834.50 | 839.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 847.55 | 837.11 | 840.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 847.55 | 837.11 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 860.00 | 844.50 | 843.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 866.70 | 851.79 | 846.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 15:15:00 | 865.90 | 867.93 | 861.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:15:00 | 867.30 | 867.93 | 861.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 861.65 | 866.93 | 862.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 861.65 | 866.93 | 862.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 861.50 | 865.85 | 862.30 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 853.80 | 859.98 | 860.39 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 866.00 | 861.18 | 860.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 874.45 | 865.74 | 863.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 13:15:00 | 925.80 | 925.94 | 911.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:30:00 | 924.40 | 925.94 | 911.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 944.20 | 943.78 | 937.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 937.75 | 943.78 | 937.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 960.50 | 963.52 | 957.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 962.70 | 963.52 | 957.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 960.30 | 962.87 | 958.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 963.80 | 962.13 | 958.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 962.50 | 963.10 | 959.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 903.45 | 950.33 | 955.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 903.45 | 950.33 | 955.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 862.10 | 873.28 | 886.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 886.75 | 872.53 | 880.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 886.75 | 872.53 | 880.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 886.75 | 872.53 | 880.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 886.75 | 872.53 | 880.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 885.00 | 875.02 | 881.22 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 15:15:00 | 891.00 | 884.86 | 884.37 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 870.80 | 882.15 | 883.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 865.90 | 875.97 | 880.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 12:15:00 | 876.05 | 872.04 | 876.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 876.05 | 872.04 | 876.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 876.05 | 872.04 | 876.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 876.05 | 872.04 | 876.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 870.95 | 871.83 | 875.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 877.00 | 871.83 | 875.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 881.10 | 874.12 | 875.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 879.90 | 874.12 | 875.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 879.80 | 875.26 | 876.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:15:00 | 880.80 | 875.26 | 876.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 889.30 | 878.07 | 877.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 897.40 | 885.67 | 881.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 894.10 | 896.88 | 891.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 894.10 | 896.88 | 891.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 895.55 | 896.61 | 891.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 896.75 | 895.87 | 892.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:15:00 | 897.20 | 895.29 | 892.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 899.55 | 895.39 | 892.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 12:15:00 | 922.60 | 926.13 | 926.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 922.60 | 926.13 | 926.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 918.40 | 923.64 | 925.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 912.70 | 911.84 | 917.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 912.70 | 911.84 | 917.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 932.45 | 913.21 | 914.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 932.50 | 913.21 | 914.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 927.45 | 916.06 | 915.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 939.10 | 926.62 | 921.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 929.30 | 930.57 | 924.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 929.30 | 930.57 | 924.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 929.30 | 930.57 | 924.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 929.30 | 930.57 | 924.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 919.40 | 928.33 | 924.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 919.40 | 928.33 | 924.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 922.50 | 927.17 | 924.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 932.55 | 926.33 | 923.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-22 09:15:00 | 1025.81 | 1017.46 | 1008.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 1067.90 | 1081.73 | 1081.77 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1084.00 | 1076.24 | 1075.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 15:15:00 | 1091.00 | 1078.98 | 1076.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 1082.40 | 1086.29 | 1082.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 1082.40 | 1086.29 | 1082.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1082.40 | 1086.29 | 1082.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1082.40 | 1086.29 | 1082.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1074.30 | 1083.89 | 1081.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1074.30 | 1083.89 | 1081.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1078.00 | 1082.71 | 1081.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1088.40 | 1082.71 | 1081.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1088.10 | 1082.64 | 1081.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:45:00 | 1087.00 | 1084.21 | 1082.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 1068.10 | 1079.37 | 1080.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1068.10 | 1079.37 | 1080.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 1053.80 | 1066.53 | 1073.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 1061.80 | 1061.20 | 1068.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 1061.80 | 1061.20 | 1068.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1066.80 | 1062.32 | 1068.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 1065.20 | 1062.32 | 1068.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1062.50 | 1062.36 | 1067.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 1065.70 | 1062.36 | 1067.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1052.00 | 1058.32 | 1064.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1060.90 | 1058.32 | 1064.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1067.50 | 1058.80 | 1062.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1067.50 | 1058.80 | 1062.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1066.90 | 1060.42 | 1062.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 1069.40 | 1060.42 | 1062.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1065.70 | 1062.06 | 1063.16 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 1067.70 | 1064.06 | 1063.93 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1059.90 | 1063.38 | 1063.75 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1070.30 | 1064.77 | 1064.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 1095.40 | 1072.04 | 1067.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1139.10 | 1147.76 | 1129.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1139.10 | 1147.76 | 1129.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1262.20 | 1270.63 | 1260.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1262.70 | 1270.63 | 1260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1257.80 | 1268.06 | 1260.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 1257.80 | 1268.06 | 1260.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1259.90 | 1266.43 | 1260.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 1254.40 | 1266.43 | 1260.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 1256.70 | 1262.18 | 1259.40 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 1244.80 | 1255.31 | 1256.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 1236.30 | 1251.51 | 1254.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 11:15:00 | 1226.70 | 1220.65 | 1229.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 12:00:00 | 1226.70 | 1220.65 | 1229.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1232.90 | 1223.10 | 1230.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 1231.80 | 1223.10 | 1230.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1233.10 | 1225.10 | 1230.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1229.50 | 1225.10 | 1230.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1235.00 | 1227.08 | 1230.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 1230.90 | 1227.08 | 1230.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1244.00 | 1231.08 | 1231.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1246.60 | 1231.08 | 1231.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1240.10 | 1232.88 | 1232.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 1254.40 | 1238.92 | 1235.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 1250.70 | 1269.78 | 1259.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 1250.70 | 1269.78 | 1259.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1250.70 | 1269.78 | 1259.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 1252.70 | 1269.78 | 1259.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1242.90 | 1264.40 | 1257.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 1240.00 | 1264.40 | 1257.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1238.30 | 1252.04 | 1253.42 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 1261.50 | 1253.98 | 1253.91 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1250.20 | 1254.15 | 1254.30 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 1266.50 | 1256.23 | 1255.20 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1224.00 | 1249.38 | 1252.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 1213.20 | 1224.76 | 1236.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 1225.20 | 1224.85 | 1235.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 1225.20 | 1224.85 | 1235.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1227.40 | 1225.93 | 1233.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:45:00 | 1227.40 | 1225.93 | 1233.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1233.70 | 1227.48 | 1233.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 1233.70 | 1227.48 | 1233.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1237.50 | 1229.48 | 1234.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1237.50 | 1229.48 | 1234.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1236.70 | 1230.93 | 1234.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1240.30 | 1230.93 | 1234.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1232.90 | 1231.77 | 1234.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 1231.00 | 1231.77 | 1234.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:15:00 | 1231.00 | 1231.63 | 1233.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1227.10 | 1227.86 | 1230.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 1238.60 | 1227.81 | 1228.25 | SL hit (close>static) qty=1.00 sl=1237.70 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 1228.80 | 1228.58 | 1228.56 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 1227.30 | 1228.33 | 1228.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 1223.80 | 1227.37 | 1227.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1227.60 | 1225.27 | 1226.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 14:15:00 | 1227.60 | 1225.27 | 1226.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1227.60 | 1225.27 | 1226.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 1227.60 | 1225.27 | 1226.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1226.50 | 1225.52 | 1226.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1233.30 | 1225.52 | 1226.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1224.50 | 1225.32 | 1226.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 1219.30 | 1222.93 | 1224.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 1229.30 | 1226.07 | 1225.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1229.30 | 1226.07 | 1225.64 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1223.20 | 1225.58 | 1225.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 1217.50 | 1223.97 | 1224.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 1199.00 | 1195.33 | 1207.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 1199.00 | 1195.33 | 1207.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1194.20 | 1196.61 | 1203.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 1187.10 | 1192.82 | 1200.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:30:00 | 1187.50 | 1191.72 | 1198.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:00:00 | 1187.30 | 1191.72 | 1198.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1128.12 | 1150.74 | 1163.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1127.74 | 1140.64 | 1153.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1127.93 | 1140.64 | 1153.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1123.40 | 1117.26 | 1126.86 | SL hit (close>ema200) qty=0.50 sl=1117.26 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1146.80 | 1131.99 | 1130.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 1149.00 | 1137.76 | 1133.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1233.00 | 1233.89 | 1214.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 1233.00 | 1233.89 | 1214.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1235.00 | 1239.33 | 1231.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 1235.00 | 1239.33 | 1231.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1237.00 | 1240.15 | 1235.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1240.60 | 1240.15 | 1235.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1243.20 | 1240.76 | 1235.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1247.70 | 1242.07 | 1236.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1247.80 | 1243.22 | 1237.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 1246.30 | 1244.33 | 1239.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 1229.50 | 1242.41 | 1239.84 | SL hit (close<static) qty=1.00 sl=1231.10 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1288.50 | 1291.80 | 1292.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 15:15:00 | 1285.20 | 1289.34 | 1290.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 1291.20 | 1289.71 | 1290.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 1291.20 | 1289.71 | 1290.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1291.20 | 1289.71 | 1290.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:15:00 | 1299.90 | 1289.71 | 1290.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1293.30 | 1290.43 | 1291.07 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 1303.90 | 1293.20 | 1292.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1307.20 | 1296.00 | 1293.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1303.00 | 1307.86 | 1303.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1303.00 | 1307.86 | 1303.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1303.00 | 1307.86 | 1303.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1303.00 | 1307.86 | 1303.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1295.50 | 1305.39 | 1302.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 1296.50 | 1305.39 | 1302.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1311.50 | 1306.61 | 1303.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1318.50 | 1308.75 | 1307.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 15:15:00 | 1302.70 | 1307.14 | 1307.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1302.70 | 1307.14 | 1307.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1292.50 | 1304.22 | 1306.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1311.10 | 1281.16 | 1285.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1311.10 | 1281.16 | 1285.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1311.10 | 1281.16 | 1285.78 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 1322.50 | 1289.43 | 1289.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1344.20 | 1317.76 | 1305.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 11:15:00 | 1338.70 | 1339.62 | 1327.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 1338.70 | 1339.62 | 1327.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1329.70 | 1337.63 | 1327.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:45:00 | 1328.30 | 1337.63 | 1327.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1334.30 | 1336.97 | 1328.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 14:15:00 | 1336.00 | 1336.97 | 1328.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1338.00 | 1334.46 | 1328.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1324.40 | 1332.45 | 1328.18 | SL hit (close<static) qty=1.00 sl=1325.40 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1321.90 | 1326.18 | 1326.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 12:15:00 | 1316.20 | 1323.09 | 1324.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 12:15:00 | 1311.10 | 1310.03 | 1315.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 1311.10 | 1310.03 | 1315.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1311.10 | 1310.03 | 1315.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 1304.70 | 1309.17 | 1314.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 1304.80 | 1308.40 | 1313.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1323.60 | 1306.51 | 1308.97 | SL hit (close>static) qty=1.00 sl=1316.80 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1327.10 | 1310.62 | 1310.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 1329.00 | 1314.30 | 1312.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1307.70 | 1320.63 | 1317.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1307.70 | 1320.63 | 1317.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1307.70 | 1320.63 | 1317.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:15:00 | 1305.10 | 1320.63 | 1317.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1304.70 | 1317.44 | 1315.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1304.50 | 1317.44 | 1315.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1300.40 | 1314.03 | 1314.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1296.60 | 1308.12 | 1311.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 12:15:00 | 1289.00 | 1288.98 | 1295.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 12:30:00 | 1288.70 | 1288.98 | 1295.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1272.80 | 1283.77 | 1290.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 1268.00 | 1283.77 | 1290.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 1268.60 | 1273.65 | 1282.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:30:00 | 1269.40 | 1272.64 | 1281.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 1266.00 | 1272.05 | 1279.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1275.40 | 1272.72 | 1279.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 1276.00 | 1272.72 | 1279.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1261.00 | 1264.71 | 1271.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 1243.00 | 1257.46 | 1267.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 1274.00 | 1254.85 | 1260.10 | SL hit (close>static) qty=1.00 sl=1272.60 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1275.90 | 1263.95 | 1263.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 1288.00 | 1274.50 | 1269.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 1364.40 | 1364.80 | 1347.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:45:00 | 1361.10 | 1364.80 | 1347.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1332.60 | 1358.39 | 1347.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 1332.60 | 1358.39 | 1347.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1325.30 | 1351.77 | 1345.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 1324.50 | 1351.77 | 1345.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 1326.80 | 1339.60 | 1341.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 1323.70 | 1332.99 | 1337.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 1334.80 | 1331.17 | 1334.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1334.80 | 1331.17 | 1334.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1334.80 | 1331.17 | 1334.69 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 1348.70 | 1338.36 | 1337.22 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1324.90 | 1337.27 | 1337.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1309.00 | 1331.62 | 1335.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 1318.70 | 1315.26 | 1322.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 1318.70 | 1315.26 | 1322.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1287.90 | 1281.95 | 1289.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1286.50 | 1281.95 | 1289.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1286.00 | 1282.76 | 1289.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 1287.60 | 1282.76 | 1289.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1286.30 | 1283.47 | 1289.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 1293.00 | 1283.47 | 1289.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1292.30 | 1285.24 | 1289.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 1292.30 | 1285.24 | 1289.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1304.30 | 1289.05 | 1290.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1304.30 | 1289.05 | 1290.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1303.30 | 1291.90 | 1291.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:15:00 | 1312.20 | 1291.90 | 1291.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1312.20 | 1295.96 | 1293.78 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1285.00 | 1295.79 | 1296.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1282.20 | 1290.01 | 1293.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1271.10 | 1271.06 | 1279.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:45:00 | 1270.70 | 1271.06 | 1279.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1284.20 | 1273.69 | 1279.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 1286.00 | 1273.69 | 1279.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1285.00 | 1275.95 | 1280.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1285.00 | 1275.95 | 1280.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1283.10 | 1277.38 | 1280.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 1282.40 | 1277.38 | 1280.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1306.00 | 1285.66 | 1283.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1306.00 | 1285.66 | 1283.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1321.50 | 1300.86 | 1292.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1329.90 | 1331.10 | 1316.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:45:00 | 1329.50 | 1331.10 | 1316.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1325.60 | 1335.32 | 1331.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1325.60 | 1335.32 | 1331.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1322.50 | 1332.76 | 1330.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1333.70 | 1332.76 | 1330.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1324.70 | 1329.97 | 1329.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1324.70 | 1329.97 | 1329.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1327.00 | 1329.38 | 1329.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 1324.20 | 1329.38 | 1329.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1324.20 | 1328.34 | 1328.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1315.80 | 1325.01 | 1327.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 1324.80 | 1323.52 | 1325.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 1324.80 | 1323.52 | 1325.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1324.80 | 1323.52 | 1325.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1325.00 | 1323.52 | 1325.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1318.90 | 1322.60 | 1325.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 1324.80 | 1322.60 | 1325.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1310.80 | 1303.15 | 1308.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1310.80 | 1303.15 | 1308.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1302.50 | 1303.02 | 1308.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:00:00 | 1297.10 | 1302.07 | 1306.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 1297.00 | 1301.66 | 1306.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 1297.70 | 1299.48 | 1304.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1292.20 | 1299.92 | 1303.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1312.90 | 1300.32 | 1302.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1312.90 | 1300.32 | 1302.18 | SL hit (close>static) qty=1.00 sl=1312.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1316.30 | 1305.29 | 1304.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1327.30 | 1309.69 | 1306.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1332.30 | 1336.22 | 1325.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1332.30 | 1336.22 | 1325.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1328.60 | 1335.21 | 1327.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1326.30 | 1335.21 | 1327.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1327.00 | 1333.57 | 1327.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 1332.80 | 1333.06 | 1327.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 1334.50 | 1333.25 | 1328.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 1333.70 | 1331.82 | 1328.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 1333.30 | 1331.24 | 1328.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1320.50 | 1329.09 | 1328.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 1320.50 | 1329.09 | 1328.18 | SL hit (close<static) qty=1.00 sl=1323.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1319.60 | 1327.19 | 1327.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1310.20 | 1319.96 | 1323.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1285.90 | 1274.79 | 1286.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1285.90 | 1274.79 | 1286.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1285.90 | 1274.79 | 1286.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1272.70 | 1276.73 | 1283.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1304.70 | 1283.79 | 1285.53 | SL hit (close>static) qty=1.00 sl=1298.50 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1315.00 | 1290.03 | 1288.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1319.60 | 1299.92 | 1293.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1326.70 | 1333.22 | 1320.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:00:00 | 1326.70 | 1333.22 | 1320.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1325.70 | 1331.72 | 1320.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 1334.40 | 1325.76 | 1320.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 1317.90 | 1324.67 | 1321.58 | SL hit (close<static) qty=1.00 sl=1318.10 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1303.10 | 1318.43 | 1319.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1290.90 | 1312.92 | 1316.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1263.60 | 1255.18 | 1276.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 1263.60 | 1255.18 | 1276.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1293.10 | 1264.47 | 1270.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 1298.90 | 1264.47 | 1270.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1251.90 | 1261.95 | 1268.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1278.60 | 1261.95 | 1268.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1177.20 | 1165.78 | 1177.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 1177.20 | 1165.78 | 1177.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1178.50 | 1168.32 | 1177.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 1179.10 | 1168.32 | 1177.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1175.90 | 1169.84 | 1177.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 1182.80 | 1169.84 | 1177.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1165.40 | 1168.95 | 1176.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 1160.00 | 1167.16 | 1174.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1124.30 | 1167.12 | 1172.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-01 12:15:00 | 1044.00 | 1151.53 | 1154.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 1181.50 | 1157.52 | 1156.56 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1176.90 | 1191.90 | 1193.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 09:15:00 | 1162.60 | 1179.15 | 1184.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 1166.90 | 1161.94 | 1170.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:00:00 | 1166.90 | 1161.94 | 1170.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1143.50 | 1127.27 | 1132.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1143.50 | 1127.27 | 1132.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1150.00 | 1131.81 | 1133.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1154.90 | 1131.81 | 1133.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1150.10 | 1135.47 | 1135.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 1171.80 | 1147.37 | 1141.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1163.20 | 1182.51 | 1168.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1163.20 | 1182.51 | 1168.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1163.20 | 1182.51 | 1168.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 1163.20 | 1182.51 | 1168.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1160.60 | 1178.13 | 1167.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 1160.60 | 1178.13 | 1167.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1163.90 | 1172.22 | 1166.74 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1143.00 | 1160.83 | 1162.47 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1172.00 | 1159.81 | 1159.78 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 1141.30 | 1158.11 | 1159.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 1136.50 | 1153.79 | 1157.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1140.50 | 1140.19 | 1148.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:15:00 | 1142.00 | 1140.19 | 1148.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1140.70 | 1140.29 | 1147.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1136.50 | 1140.29 | 1147.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1136.30 | 1134.31 | 1140.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1079.67 | 1101.71 | 1115.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1079.48 | 1101.71 | 1115.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 1049.10 | 1048.79 | 1070.64 | SL hit (close>ema200) qty=0.50 sl=1048.79 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 1020.30 | 1001.31 | 1001.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1043.90 | 1020.44 | 1012.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1052.70 | 1057.71 | 1039.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1052.70 | 1057.71 | 1039.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1052.70 | 1057.71 | 1039.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 1057.00 | 1057.11 | 1040.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 1057.50 | 1057.11 | 1040.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 1056.40 | 1056.57 | 1043.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1055.10 | 1050.01 | 1043.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1066.90 | 1052.08 | 1045.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1001.80 | 1043.89 | 1045.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1001.80 | 1043.89 | 1045.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 993.70 | 1033.85 | 1040.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1009.90 | 1007.47 | 1021.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 1022.30 | 1011.02 | 1020.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1022.30 | 1011.02 | 1020.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1022.30 | 1011.02 | 1020.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1031.10 | 1015.04 | 1021.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1035.80 | 1015.04 | 1021.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1034.20 | 1018.87 | 1022.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1034.20 | 1018.87 | 1022.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1051.50 | 1029.94 | 1027.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1056.80 | 1035.31 | 1029.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1033.50 | 1050.33 | 1041.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1033.50 | 1050.33 | 1041.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1033.50 | 1050.33 | 1041.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1033.50 | 1050.33 | 1041.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1021.00 | 1044.46 | 1039.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1021.00 | 1044.46 | 1039.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1019.50 | 1034.92 | 1036.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1010.40 | 1027.23 | 1032.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 993.60 | 982.40 | 1000.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 993.60 | 982.40 | 1000.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 993.60 | 982.40 | 1000.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 972.50 | 995.01 | 1000.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 1007.55 | 991.43 | 994.51 | SL hit (close>static) qty=1.00 sl=1001.85 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1019.00 | 996.85 | 995.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1025.00 | 1002.48 | 998.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1095.85 | 1105.28 | 1079.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 1097.20 | 1105.28 | 1079.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1096.25 | 1111.53 | 1099.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1101.80 | 1110.17 | 1099.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1102.70 | 1107.46 | 1100.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1103.70 | 1107.46 | 1100.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1148.90 | 1162.10 | 1162.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1148.90 | 1162.10 | 1162.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 1146.90 | 1159.06 | 1160.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 14:15:00 | 1157.40 | 1155.09 | 1157.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 1157.40 | 1155.09 | 1157.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1157.40 | 1155.09 | 1157.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 1157.40 | 1155.09 | 1157.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1164.00 | 1156.87 | 1158.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1162.30 | 1156.87 | 1158.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1158.85 | 1157.27 | 1158.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 1161.95 | 1157.27 | 1158.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1148.45 | 1155.50 | 1157.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:30:00 | 1146.15 | 1154.59 | 1157.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 1146.10 | 1152.18 | 1155.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 1147.35 | 1152.18 | 1155.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 15:00:00 | 1146.10 | 1150.96 | 1154.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1153.00 | 1151.37 | 1154.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1061.60 | 1151.37 | 1154.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1088.84 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1088.79 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1089.98 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1088.79 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 1140.70 | 1134.62 | 1143.90 | SL hit (close>ema200) qty=0.50 sl=1134.62 alert=retest2 |

### Cycle 63 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1108.40 | 1106.63 | 1106.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1173.20 | 1120.48 | 1112.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1186.60 | 1191.98 | 1168.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 1186.60 | 1191.98 | 1168.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 11:45:00 | 853.80 | 2025-05-14 09:15:00 | 836.45 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-13 12:15:00 | 853.30 | 2025-05-14 09:15:00 | 836.45 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-13 14:45:00 | 859.50 | 2025-05-14 09:15:00 | 836.45 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-06-11 09:15:00 | 963.80 | 2025-06-12 09:15:00 | 903.45 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest2 | 2025-06-11 10:30:00 | 962.50 | 2025-06-12 09:15:00 | 903.45 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest2 | 2025-06-25 15:15:00 | 896.75 | 2025-07-07 12:15:00 | 922.60 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2025-06-26 10:15:00 | 897.20 | 2025-07-07 12:15:00 | 922.60 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-06-26 11:15:00 | 899.55 | 2025-07-07 12:15:00 | 922.60 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest2 | 2025-07-11 14:15:00 | 932.55 | 2025-07-22 09:15:00 | 1025.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1088.40 | 2025-08-05 09:15:00 | 1068.10 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1088.10 | 2025-08-05 09:15:00 | 1068.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-08-04 13:45:00 | 1087.00 | 2025-08-05 09:15:00 | 1068.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-11 11:15:00 | 1231.00 | 2025-09-15 11:15:00 | 1238.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-11 12:15:00 | 1231.00 | 2025-09-15 11:15:00 | 1238.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-12 10:45:00 | 1227.10 | 2025-09-15 11:15:00 | 1238.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-18 09:30:00 | 1219.30 | 2025-09-18 14:15:00 | 1229.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1187.10 | 2025-09-26 09:15:00 | 1128.12 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-09-23 13:30:00 | 1187.50 | 2025-09-26 14:15:00 | 1127.74 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-23 14:00:00 | 1187.30 | 2025-09-26 14:15:00 | 1127.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1187.10 | 2025-09-30 14:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2025-09-23 13:30:00 | 1187.50 | 2025-09-30 14:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-09-23 14:00:00 | 1187.30 | 2025-09-30 14:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.38% |
| BUY | retest2 | 2025-10-13 10:30:00 | 1247.70 | 2025-10-14 09:15:00 | 1229.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-13 12:00:00 | 1247.80 | 2025-10-14 09:15:00 | 1229.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-13 13:45:00 | 1246.30 | 2025-10-14 09:15:00 | 1229.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-14 14:45:00 | 1248.20 | 2025-10-24 12:15:00 | 1288.50 | STOP_HIT | 1.00 | 3.23% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1258.70 | 2025-10-24 12:15:00 | 1288.50 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1318.50 | 2025-10-31 15:15:00 | 1302.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-10 14:15:00 | 1336.00 | 2025-11-11 09:15:00 | 1324.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-11 09:15:00 | 1338.00 | 2025-11-11 09:15:00 | 1324.40 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1336.30 | 2025-11-12 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-13 14:45:00 | 1304.70 | 2025-11-17 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-11-14 10:00:00 | 1304.80 | 2025-11-17 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-11-21 10:15:00 | 1268.00 | 2025-11-26 10:15:00 | 1274.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-11-21 13:30:00 | 1268.60 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-21 14:30:00 | 1269.40 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-24 09:30:00 | 1266.00 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-25 11:30:00 | 1243.00 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-18 14:15:00 | 1282.40 | 2025-12-19 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-12-31 14:00:00 | 1297.10 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-31 15:15:00 | 1297.00 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-01 09:30:00 | 1297.70 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1292.20 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-01-06 12:15:00 | 1332.80 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-01-06 12:45:00 | 1334.50 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-06 15:00:00 | 1333.70 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1333.30 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1272.70 | 2026-01-14 09:15:00 | 1304.70 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-01-19 15:00:00 | 1334.40 | 2026-01-20 10:15:00 | 1317.90 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-29 12:00:00 | 1160.00 | 2026-02-01 12:15:00 | 1044.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1124.30 | 2026-02-01 12:15:00 | 1068.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1124.30 | 2026-02-01 12:15:00 | 1227.30 | STOP_HIT | 0.50 | -9.16% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1136.50 | 2026-03-02 09:15:00 | 1079.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:45:00 | 1136.30 | 2026-03-02 09:15:00 | 1079.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1136.50 | 2026-03-04 13:15:00 | 1049.10 | STOP_HIT | 0.50 | 7.69% |
| SELL | retest2 | 2026-02-26 09:45:00 | 1136.30 | 2026-03-04 13:15:00 | 1049.10 | STOP_HIT | 0.50 | 7.67% |
| BUY | retest2 | 2026-03-19 10:30:00 | 1057.00 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2026-03-19 11:15:00 | 1057.50 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.27% |
| BUY | retest2 | 2026-03-19 12:45:00 | 1056.40 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.17% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1055.10 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 972.50 | 2026-04-02 14:15:00 | 1007.55 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-04-06 11:00:00 | 978.60 | 2026-04-06 12:15:00 | 1019.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1101.80 | 2026-04-23 09:15:00 | 1148.90 | STOP_HIT | 1.00 | 4.27% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1102.70 | 2026-04-23 09:15:00 | 1148.90 | STOP_HIT | 1.00 | 4.19% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1103.70 | 2026-04-23 09:15:00 | 1148.90 | STOP_HIT | 1.00 | 4.10% |
| SELL | retest2 | 2026-04-24 11:30:00 | 1146.15 | 2026-04-27 09:15:00 | 1088.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 13:30:00 | 1146.10 | 2026-04-27 09:15:00 | 1088.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 14:00:00 | 1147.35 | 2026-04-27 09:15:00 | 1089.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 15:00:00 | 1146.10 | 2026-04-27 09:15:00 | 1088.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 11:30:00 | 1146.15 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2026-04-24 13:30:00 | 1146.10 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2026-04-24 14:00:00 | 1147.35 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.58% |
| SELL | retest2 | 2026-04-24 15:00:00 | 1146.10 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2026-04-27 09:15:00 | 1061.60 | 2026-05-06 14:15:00 | 1108.40 | STOP_HIT | 1.00 | -4.41% |
