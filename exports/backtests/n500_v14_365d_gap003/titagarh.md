# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 840.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 0
- **Avg / median % per leg:** -0.77% / -1.17%
- **Sum % (uncompounded):** -12.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.77% | -12.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.77% | -12.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.77% | -12.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 938.80 | 825.84 | 825.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 942.75 | 863.95 | 846.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 881.60 | 884.78 | 861.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 873.90 | 884.78 | 861.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 862.95 | 884.52 | 863.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 862.95 | 884.52 | 863.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 855.70 | 884.23 | 863.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 853.00 | 884.23 | 863.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 865.00 | 881.22 | 863.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 872.00 | 880.82 | 863.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 09:15:00 | 959.20 | 890.14 | 870.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 872.20 | 916.89 | 901.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:45:00 | 867.50 | 915.91 | 901.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 867.50 | 915.91 | 901.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 854.95 | 908.39 | 898.77 | SL hit (close<static) qty=1.00 sl=858.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 854.95 | 908.39 | 898.77 | SL hit (close<static) qty=1.00 sl=858.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 854.95 | 908.39 | 898.77 | SL hit (close<static) qty=1.00 sl=858.20 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-08 09:15:00 | 834.75 | 890.39 | 890.55 | min_gap filter: gap=0.020% < 0.030% |
| TREND_RESET | 2025-08-08 09:15:00 | 834.75 | 890.39 | 890.55 | EMA inversion without crossover edge (EMA200=890.39 EMA400=890.55) — end cycle |

### Cycle 2 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 934.65 | 873.48 | 873.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 937.70 | 874.12 | 873.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 887.00 | 895.49 | 886.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 882.05 | 895.27 | 886.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 882.05 | 895.27 | 886.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 880.20 | 895.12 | 886.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 893.70 | 894.98 | 886.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 884.50 | 894.46 | 886.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 872.50 | 894.09 | 885.99 | SL hit (close<static) qty=1.00 sl=876.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 872.50 | 894.09 | 885.99 | SL hit (close<static) qty=1.00 sl=876.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:15:00 | 886.00 | 892.27 | 885.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:00:00 | 883.00 | 892.05 | 885.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 886.00 | 891.99 | 885.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 890.95 | 891.88 | 885.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 882.95 | 891.78 | 885.49 | SL hit (close<static) qty=1.00 sl=883.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 892.50 | 891.55 | 885.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 890.55 | 894.87 | 888.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 14:15:00 | 882.50 | 894.38 | 888.32 | SL hit (close<static) qty=1.00 sl=883.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 14:15:00 | 882.50 | 894.38 | 888.32 | SL hit (close<static) qty=1.00 sl=883.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 890.35 | 894.26 | 888.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 889.55 | 894.12 | 888.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 889.55 | 894.12 | 888.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 890.00 | 894.08 | 888.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 890.00 | 894.08 | 888.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 889.60 | 893.98 | 888.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 889.05 | 893.98 | 888.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 886.50 | 893.91 | 888.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 886.10 | 893.91 | 888.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 882.10 | 893.79 | 888.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 882.10 | 893.79 | 888.37 | SL hit (close<static) qty=1.00 sl=883.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 868.15 | 892.84 | 888.07 | SL hit (close<static) qty=1.00 sl=876.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 868.15 | 892.84 | 888.07 | SL hit (close<static) qty=1.00 sl=876.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:30:00 | 889.00 | 891.00 | 887.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:45:00 | 887.00 | 890.96 | 887.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 878.75 | 890.62 | 887.32 | SL hit (close<static) qty=1.00 sl=879.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 878.75 | 890.62 | 887.32 | SL hit (close<static) qty=1.00 sl=879.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 888.55 | 889.50 | 886.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 906.00 | 891.93 | 888.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 888.65 | 893.02 | 889.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 890.30 | 893.02 | 889.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 889.05 | 892.98 | 889.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:15:00 | 885.00 | 892.98 | 889.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 883.50 | 892.89 | 889.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 883.50 | 892.89 | 889.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 878.55 | 892.74 | 889.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 878.55 | 892.74 | 889.24 | SL hit (close<static) qty=1.00 sl=879.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 878.55 | 892.74 | 889.24 | SL hit (close<static) qty=1.00 sl=879.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 878.55 | 892.74 | 889.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| CROSSOVER_SKIP | 2025-11-10 11:15:00 | 846.65 | 885.87 | 885.98 | min_gap filter: gap=0.013% < 0.030% |
| TREND_RESET | 2025-11-10 11:15:00 | 846.65 | 885.87 | 885.98 | EMA inversion without crossover edge (EMA200=885.87 EMA400=885.98) — end cycle |

### Cycle 3 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 863.85 | 725.43 | 724.75 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 15:15:00 | 872.00 | 2025-06-27 09:15:00 | 959.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 12:00:00 | 872.20 | 2025-08-01 09:15:00 | 854.95 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-29 13:45:00 | 867.50 | 2025-08-01 09:15:00 | 854.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-29 14:15:00 | 867.50 | 2025-08-01 09:15:00 | 854.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-29 09:15:00 | 893.70 | 2025-09-30 11:15:00 | 872.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-30 09:45:00 | 884.50 | 2025-09-30 11:15:00 | 872.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-01 15:15:00 | 886.00 | 2025-10-06 09:15:00 | 882.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-10-03 11:00:00 | 883.00 | 2025-10-14 14:15:00 | 882.50 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-03 15:15:00 | 890.95 | 2025-10-14 14:15:00 | 882.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-06 14:30:00 | 892.50 | 2025-10-16 12:15:00 | 882.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-10-13 12:30:00 | 890.55 | 2025-10-17 12:15:00 | 868.15 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-10-15 09:15:00 | 890.35 | 2025-10-17 12:15:00 | 868.15 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-10-23 09:30:00 | 889.00 | 2025-10-23 14:15:00 | 878.75 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-23 10:45:00 | 887.00 | 2025-10-23 14:15:00 | 878.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-27 11:45:00 | 888.55 | 2025-11-04 13:15:00 | 878.55 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-11-03 09:15:00 | 906.00 | 2025-11-04 13:15:00 | 878.55 | STOP_HIT | 1.00 | -3.03% |
