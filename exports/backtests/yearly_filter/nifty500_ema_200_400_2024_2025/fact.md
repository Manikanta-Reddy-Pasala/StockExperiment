# Fertilisers and Chemicals Travancore Ltd. (FACT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 902.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 20 |
| TARGET_HIT | 14 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 26
- **Target hits / Stop hits / Partials:** 13 / 34 / 20
- **Avg / median % per leg:** 2.37% / 3.35%
- **Sum % (uncompounded):** 158.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 3 | 15.8% | 3 | 16 | 0 | -1.02% | -19.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 3 | 15.8% | 3 | 16 | 0 | -1.02% | -19.4% |
| SELL (all) | 48 | 38 | 79.2% | 10 | 18 | 20 | 3.71% | 178.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 48 | 38 | 79.2% | 10 | 18 | 20 | 3.71% | 178.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 67 | 41 | 61.2% | 13 | 34 | 20 | 2.37% | 158.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 779.05 | 702.16 | 701.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 786.30 | 705.74 | 703.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 999.90 | 1002.89 | 929.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 12:15:00 | 940.90 | 1000.75 | 931.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 940.90 | 1000.75 | 931.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 13:00:00 | 940.90 | 1000.75 | 931.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 936.00 | 998.92 | 931.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 956.85 | 998.92 | 931.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 938.20 | 995.73 | 932.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-09 09:15:00 | 1032.02 | 991.23 | 934.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 15:15:00 | 882.90 | 956.27 | 956.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 858.20 | 937.58 | 946.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 891.90 | 888.49 | 913.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 14:45:00 | 889.90 | 888.49 | 913.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 917.95 | 888.84 | 913.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 887.70 | 889.81 | 913.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 10:15:00 | 887.55 | 889.81 | 913.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 843.32 | 886.28 | 909.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 843.17 | 886.28 | 909.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 866.35 | 862.41 | 889.74 | SL hit (close>ema200) qty=0.50 sl=862.41 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1025.95 | 909.91 | 909.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 1028.90 | 912.25 | 910.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 951.85 | 960.86 | 939.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 951.85 | 960.86 | 939.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 938.00 | 960.36 | 939.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 931.85 | 960.36 | 939.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 941.00 | 960.17 | 939.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 944.25 | 960.17 | 939.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 11:15:00 | 924.15 | 959.81 | 939.74 | SL hit (close<static) qty=1.00 sl=930.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 10:15:00 | 899.65 | 940.17 | 940.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 11:15:00 | 895.75 | 939.73 | 939.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 10:15:00 | 982.20 | 935.25 | 937.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 982.20 | 935.25 | 937.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 982.20 | 935.25 | 937.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 982.20 | 935.25 | 937.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 941.35 | 935.31 | 937.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 913.75 | 935.10 | 937.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 912.95 | 934.88 | 937.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 868.06 | 922.49 | 930.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 867.30 | 922.49 | 930.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 822.38 | 920.43 | 928.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 830.00 | 745.50 | 745.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 834.00 | 747.97 | 746.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 967.40 | 971.67 | 906.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 967.40 | 971.67 | 906.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 916.70 | 957.71 | 918.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 916.70 | 957.71 | 918.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 917.85 | 957.31 | 918.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 946.55 | 952.37 | 918.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 923.95 | 952.23 | 934.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 925.40 | 950.32 | 934.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 911.70 | 949.94 | 934.13 | SL hit (close<static) qty=1.00 sl=915.45 alert=retest2 |

### Cycle 6 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 898.00 | 955.29 | 955.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 15:15:00 | 897.55 | 954.71 | 955.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 14:15:00 | 903.00 | 902.78 | 918.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 903.00 | 902.78 | 918.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 913.90 | 902.91 | 918.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 896.40 | 904.36 | 916.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 14:15:00 | 851.58 | 891.98 | 906.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 13:15:00 | 806.76 | 879.16 | 898.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 870.00 | 807.96 | 807.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 931.50 | 809.19 | 808.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-21 13:00:00 | 704.40 | 2024-05-24 09:15:00 | 732.50 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2024-05-22 09:30:00 | 697.20 | 2024-05-24 09:15:00 | 732.50 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2024-05-28 09:30:00 | 703.20 | 2024-06-04 09:15:00 | 668.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 12:15:00 | 701.70 | 2024-06-04 09:15:00 | 666.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 13:45:00 | 702.05 | 2024-06-04 09:15:00 | 666.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 13:30:00 | 701.70 | 2024-06-04 09:15:00 | 666.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 13:45:00 | 700.00 | 2024-06-04 10:15:00 | 665.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:30:00 | 703.20 | 2024-06-04 12:15:00 | 632.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-28 12:15:00 | 701.70 | 2024-06-04 12:15:00 | 631.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-28 13:45:00 | 702.05 | 2024-06-04 12:15:00 | 631.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-29 13:30:00 | 701.70 | 2024-06-04 12:15:00 | 631.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 13:45:00 | 700.00 | 2024-06-04 12:15:00 | 630.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 956.85 | 2024-08-09 09:15:00 | 1032.02 | TARGET_HIT | 1.00 | 7.86% |
| BUY | retest2 | 2024-08-07 09:15:00 | 938.20 | 2024-08-14 09:15:00 | 927.65 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-08-13 14:30:00 | 941.35 | 2024-08-14 09:15:00 | 927.65 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-08-16 10:30:00 | 938.00 | 2024-08-16 13:15:00 | 930.75 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-08-20 12:45:00 | 933.20 | 2024-08-22 09:15:00 | 1026.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 887.70 | 2024-11-13 09:15:00 | 843.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 10:15:00 | 887.55 | 2024-11-13 09:15:00 | 843.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 887.70 | 2024-11-27 09:15:00 | 866.35 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-11-08 10:15:00 | 887.55 | 2024-11-27 09:15:00 | 866.35 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2024-11-29 09:30:00 | 889.60 | 2024-11-29 11:15:00 | 943.05 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest2 | 2024-12-23 11:15:00 | 944.25 | 2024-12-23 11:15:00 | 924.15 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-24 09:45:00 | 967.95 | 2025-01-06 14:15:00 | 926.95 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-01-07 10:30:00 | 966.00 | 2025-01-13 09:15:00 | 912.00 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest2 | 2025-01-10 11:15:00 | 947.10 | 2025-01-13 09:15:00 | 912.00 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-01-21 09:15:00 | 991.50 | 2025-01-22 09:15:00 | 933.20 | STOP_HIT | 1.00 | -5.88% |
| BUY | retest2 | 2025-01-21 11:45:00 | 962.10 | 2025-01-22 09:15:00 | 933.20 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-01-23 09:45:00 | 965.10 | 2025-01-27 09:15:00 | 913.80 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2025-01-24 10:15:00 | 961.00 | 2025-01-27 09:15:00 | 913.80 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2025-02-01 13:00:00 | 913.75 | 2025-02-10 13:15:00 | 868.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 15:15:00 | 912.95 | 2025-02-10 13:15:00 | 867.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 13:00:00 | 913.75 | 2025-02-11 09:15:00 | 822.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 15:15:00 | 912.95 | 2025-02-11 09:15:00 | 821.66 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-15 09:15:00 | 946.55 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-08-07 14:00:00 | 923.95 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-08-11 09:15:00 | 925.40 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-12 09:15:00 | 930.90 | 2025-08-21 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 15:15:00 | 954.00 | 2025-09-26 11:15:00 | 942.05 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-02 09:30:00 | 955.40 | 2025-09-26 11:15:00 | 942.05 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-21 09:15:00 | 896.40 | 2025-12-02 14:15:00 | 851.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 896.40 | 2025-12-08 13:15:00 | 806.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-16 13:30:00 | 899.15 | 2025-12-18 09:15:00 | 854.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 14:15:00 | 897.60 | 2025-12-18 09:15:00 | 852.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 15:00:00 | 897.60 | 2025-12-18 09:15:00 | 852.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 13:30:00 | 899.15 | 2025-12-19 15:15:00 | 869.00 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-12-16 14:15:00 | 897.60 | 2025-12-19 15:15:00 | 869.00 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-12-16 15:00:00 | 897.60 | 2025-12-19 15:15:00 | 869.00 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-12-24 14:15:00 | 876.80 | 2025-12-26 09:15:00 | 894.75 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-01-06 15:15:00 | 875.10 | 2026-01-08 09:15:00 | 890.75 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-16 15:15:00 | 831.87 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-01-09 11:00:00 | 875.65 | 2026-01-19 09:15:00 | 829.87 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-20 15:15:00 | 788.09 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-01-09 11:00:00 | 875.65 | 2026-01-21 09:15:00 | 786.19 | TARGET_HIT | 0.50 | 10.22% |
| SELL | retest2 | 2026-03-17 09:15:00 | 788.85 | 2026-03-17 10:15:00 | 826.20 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-03-19 13:15:00 | 797.35 | 2026-03-23 09:15:00 | 758.95 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-23 10:15:00 | 757.48 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-23 10:15:00 | 758.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 13:15:00 | 797.35 | 2026-03-24 12:15:00 | 789.20 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-24 12:15:00 | 789.20 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-24 12:15:00 | 789.20 | STOP_HIT | 0.50 | 1.10% |
| SELL | retest2 | 2026-03-27 10:30:00 | 781.75 | 2026-03-30 15:15:00 | 742.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 11:00:00 | 781.45 | 2026-03-30 15:15:00 | 742.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 10:30:00 | 781.75 | 2026-04-01 11:15:00 | 792.95 | STOP_HIT | 0.50 | -1.43% |
| SELL | retest2 | 2026-03-27 11:00:00 | 781.45 | 2026-04-01 11:15:00 | 792.95 | STOP_HIT | 0.50 | -1.47% |
| SELL | retest2 | 2026-03-30 09:15:00 | 774.55 | 2026-04-08 12:15:00 | 829.15 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2026-04-01 14:30:00 | 780.80 | 2026-04-08 12:15:00 | 829.15 | STOP_HIT | 1.00 | -6.19% |
