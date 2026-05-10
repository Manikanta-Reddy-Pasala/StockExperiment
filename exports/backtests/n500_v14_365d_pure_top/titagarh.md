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
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 8 |
| TARGET_HIT | 9 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 21
- **Target hits / Stop hits / Partials:** 9 / 21 / 8
- **Avg / median % per leg:** 2.30% / -0.34%
- **Sum % (uncompounded):** 87.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.77% | -12.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.77% | -12.4% |
| SELL (all) | 22 | 16 | 72.7% | 8 | 6 | 8 | 4.54% | 99.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 16 | 72.7% | 8 | 6 | 8 | 4.54% | 99.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 17 | 44.7% | 9 | 21 | 8 | 2.30% | 87.5% |

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

### Cycle 2 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 834.75 | 890.39 | 890.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 826.55 | 889.22 | 889.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 858.90 | 857.61 | 871.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 858.90 | 857.61 | 871.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 853.00 | 857.53 | 871.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 849.35 | 857.41 | 870.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 849.50 | 857.83 | 870.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 848.25 | 858.39 | 870.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 849.35 | 858.26 | 870.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 874.00 | 852.54 | 863.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 876.35 | 852.54 | 863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 879.10 | 852.80 | 863.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 878.00 | 852.80 | 863.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 886.45 | 853.93 | 864.30 | SL hit (close>static) qty=1.00 sl=885.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 886.45 | 853.93 | 864.30 | SL hit (close>static) qty=1.00 sl=885.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 886.45 | 853.93 | 864.30 | SL hit (close>static) qty=1.00 sl=885.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 886.45 | 853.93 | 864.30 | SL hit (close>static) qty=1.00 sl=885.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-16 13:15:00)

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

### Cycle 4 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 846.65 | 885.87 | 885.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 846.00 | 885.48 | 885.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 886.95 | 882.66 | 884.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 887.35 | 882.71 | 884.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 881.40 | 882.78 | 884.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 893.80 | 880.75 | 883.16 | SL hit (close>static) qty=1.00 sl=891.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 882.00 | 880.75 | 883.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 883.25 | 880.85 | 883.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 883.10 | 880.85 | 883.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 883.50 | 880.88 | 883.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 885.05 | 880.88 | 883.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 881.80 | 880.89 | 883.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 880.30 | 880.89 | 883.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 880.00 | 880.89 | 883.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 09:15:00 | 839.09 | 875.04 | 879.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 09:15:00 | 838.94 | 875.04 | 879.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 14:15:00 | 837.90 | 873.39 | 878.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 14:15:00 | 836.28 | 873.39 | 878.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 15:15:00 | 836.00 | 873.02 | 878.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-03 13:15:00 | 793.80 | 857.22 | 868.99 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-03 13:15:00 | 794.93 | 857.22 | 868.99 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-03 13:15:00 | 794.79 | 857.22 | 868.99 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-03 13:15:00 | 792.27 | 857.22 | 868.99 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-03 14:15:00 | 792.00 | 856.59 | 868.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 878.20 | 831.14 | 842.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 888.45 | 832.19 | 843.23 | SL hit (close>static) qty=1.00 sl=883.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 880.70 | 844.51 | 848.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 854.90 | 846.87 | 849.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 855.80 | 846.87 | 849.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 853.30 | 846.94 | 849.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 851.00 | 846.94 | 849.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:15:00 | 836.66 | 846.78 | 849.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 792.63 | 842.87 | 847.20 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 839.20 | 814.96 | 828.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 839.20 | 814.96 | 828.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 829.15 | 815.10 | 828.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 845.60 | 815.10 | 828.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 813.50 | 815.08 | 828.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 789.75 | 815.08 | 828.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:00:00 | 798.75 | 814.92 | 828.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 750.26 | 798.34 | 814.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 758.81 | 798.34 | 814.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-25 15:15:00 | 718.88 | 776.10 | 798.10 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-26 14:15:00 | 710.77 | 772.54 | 795.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-05 12:15:00)

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
| SELL | retest2 | 2025-08-22 10:45:00 | 849.35 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-08-26 09:30:00 | 849.50 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-08-28 09:15:00 | 848.25 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-08-28 11:15:00 | 849.35 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.37% |
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
| SELL | retest2 | 2025-11-12 13:45:00 | 881.40 | 2025-11-17 09:15:00 | 893.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-17 09:30:00 | 882.00 | 2025-11-25 09:15:00 | 839.09 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-11-17 11:30:00 | 883.25 | 2025-11-25 09:15:00 | 838.94 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-11-17 12:00:00 | 883.10 | 2025-11-25 14:15:00 | 837.90 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2025-11-17 14:15:00 | 880.30 | 2025-11-25 14:15:00 | 836.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 15:15:00 | 880.00 | 2025-11-25 15:15:00 | 836.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 09:30:00 | 882.00 | 2025-12-03 13:15:00 | 793.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 883.25 | 2025-12-03 13:15:00 | 794.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 12:00:00 | 883.10 | 2025-12-03 13:15:00 | 794.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 14:15:00 | 880.30 | 2025-12-03 13:15:00 | 792.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 15:15:00 | 880.00 | 2025-12-03 14:15:00 | 792.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-30 15:00:00 | 878.20 | 2025-12-31 09:15:00 | 888.45 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-05 14:00:00 | 880.70 | 2026-01-08 10:15:00 | 836.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 880.70 | 2026-01-12 09:15:00 | 792.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 789.75 | 2026-02-16 09:15:00 | 750.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 13:00:00 | 798.75 | 2026-02-16 09:15:00 | 758.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 789.75 | 2026-02-25 15:15:00 | 718.88 | TARGET_HIT | 0.50 | 8.97% |
| SELL | retest2 | 2026-02-01 13:00:00 | 798.75 | 2026-02-26 14:15:00 | 710.77 | TARGET_HIT | 0.50 | 11.01% |
