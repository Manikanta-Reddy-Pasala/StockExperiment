# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 973.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 16 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 12
- **Target hits / Stop hits / Partials:** 0 / 13 / 0
- **Avg / median % per leg:** -3.17% / -2.08%
- **Sum % (uncompounded):** -41.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 0 | 13 | 0 | -3.17% | -41.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.89% | -5.7% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 0 | 10 | 0 | -3.56% | -35.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.89% | -5.7% |
| retest2 (combined) | 10 | 1 | 10.0% | 0 | 10 | 0 | -3.56% | -35.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 730.67 | 686.39 | 686.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 735.53 | 687.24 | 686.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 729.67 | 739.73 | 721.15 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 729.67 | 739.73 | 721.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 729.67 | 739.73 | 721.15 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-03 11:15:00 | 749.44 | 693.72 | 693.88 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 2 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 743.62 | 694.22 | 694.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 755.91 | 696.16 | 695.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 849.50 | 859.85 | 827.26 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-04-04 10:15:00 | 876.18 | 859.92 | 828.42 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 11:15:00 | 873.79 | 860.06 | 828.64 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 875.43 | 859.77 | 830.33 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:15:00 | 876.70 | 859.94 | 830.56 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 14:15:00 | 872.08 | 861.55 | 832.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 15:15:00 | 874.50 | 861.68 | 833.17 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 861.35 | 889.77 | 859.13 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 858.50 | 889.17 | 859.13 | SL hit (close<ema400) qty=1.00 sl=859.13 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 858.50 | 889.17 | 859.13 | SL hit (close<ema400) qty=1.00 sl=859.13 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 858.50 | 889.17 | 859.13 | SL hit (close<ema400) qty=1.00 sl=859.13 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-05 09:15:00 | 896.25 | 887.44 | 859.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 10:15:00 | 899.30 | 887.56 | 860.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 10:15:00 | 889.25 | 887.24 | 861.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 11:15:00 | 888.70 | 887.26 | 862.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-07 12:15:00 | 890.00 | 887.28 | 862.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 894.80 | 887.36 | 862.31 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 894.00 | 886.45 | 863.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 894.00 | 886.53 | 864.05 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 895.30 | 905.88 | 887.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 893.00 | 905.76 | 887.54 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 899.10 | 905.32 | 887.68 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 932.00 | 905.59 | 887.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 938.30 | 905.91 | 888.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 902.00 | 918.82 | 901.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-20 11:15:00 | 901.00 | 918.64 | 901.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-20 14:15:00 | 905.50 | 918.11 | 901.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 901.50 | 917.94 | 901.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 904.00 | 917.61 | 901.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 912.00 | 917.56 | 901.40 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 884.50 | 930.61 | 919.82 | SL hit (close<static) qty=1.00 sl=887.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 884.50 | 930.61 | 919.82 | SL hit (close<static) qty=1.00 sl=887.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 884.50 | 930.61 | 919.82 | SL hit (close<static) qty=1.00 sl=887.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 853.80 | 902.51 | 906.72 | SL hit (close<static) qty=1.00 sl=855.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 853.80 | 902.51 | 906.72 | SL hit (close<static) qty=1.00 sl=855.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 853.80 | 902.51 | 906.72 | SL hit (close<static) qty=1.00 sl=855.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 853.80 | 902.51 | 906.72 | SL hit (close<static) qty=1.00 sl=855.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 912.55 | 896.66 | 903.30 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 892.80 | 897.22 | 903.39 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-20 12:15:00 | 887.50 | 897.01 | 902.98 | SL hit (close<static) qty=1.00 sl=887.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 936.50 | 894.07 | 899.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 943.20 | 894.56 | 899.79 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 948.95 | 904.73 | 904.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 948.95 | 904.73 | 904.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 961.00 | 905.71 | 905.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-05 10:15:00 | 1059.50 | 1024.98 | 1012.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 1050.20 | 1025.23 | 1012.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 1002.70 | 1024.39 | 1013.92 | SL hit (close<static) qty=1.00 sl=1005.30 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1037.60 | 983.25 | 983.13 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-04-04 11:15:00 | 873.79 | 2025-04-30 11:15:00 | 858.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest1 | 2025-04-08 10:15:00 | 876.70 | 2025-04-30 11:15:00 | 858.50 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest1 | 2025-04-09 15:15:00 | 874.50 | 2025-04-30 11:15:00 | 858.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-05 10:15:00 | 899.30 | 2025-07-28 13:15:00 | 884.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-07 13:15:00 | 894.80 | 2025-07-28 13:15:00 | 884.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-05-12 10:15:00 | 894.00 | 2025-07-28 13:15:00 | 884.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-05 12:15:00 | 893.00 | 2025-08-12 14:15:00 | 853.80 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-06-06 11:15:00 | 938.30 | 2025-08-12 14:15:00 | 853.80 | STOP_HIT | 1.00 | -9.01% |
| BUY | retest2 | 2025-06-20 15:15:00 | 901.50 | 2025-08-12 14:15:00 | 853.80 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest2 | 2025-06-23 11:15:00 | 912.00 | 2025-08-12 14:15:00 | 853.80 | STOP_HIT | 1.00 | -6.38% |
| BUY | retest2 | 2025-08-18 10:15:00 | 912.55 | 2025-08-20 12:15:00 | 887.50 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-09-04 10:15:00 | 943.20 | 2025-09-09 14:15:00 | 948.95 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-12-05 11:15:00 | 1050.20 | 2025-12-11 12:15:00 | 1002.70 | STOP_HIT | 1.00 | -4.52% |
