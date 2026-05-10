# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1282.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -1.74% / -1.41%
- **Sum % (uncompounded):** -12.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.74% | -12.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.74% | -12.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.74% | -12.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 897.80 | 781.14 | 780.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 901.60 | 786.55 | 783.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 907.00 | 910.30 | 878.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:15:00 | 899.20 | 910.30 | 878.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 892.50 | 912.59 | 889.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 892.50 | 912.59 | 889.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 884.00 | 912.30 | 889.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 878.00 | 912.30 | 889.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 881.00 | 911.99 | 889.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 876.45 | 911.99 | 889.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 871.55 | 911.59 | 889.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 871.55 | 911.59 | 889.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 897.05 | 911.78 | 892.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 888.85 | 911.78 | 892.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 887.80 | 911.54 | 892.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 882.45 | 911.54 | 892.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 875.10 | 911.18 | 892.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 875.10 | 911.18 | 892.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 886.00 | 910.15 | 892.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 887.75 | 910.15 | 892.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 879.20 | 909.31 | 892.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 882.20 | 909.31 | 892.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 883.65 | 895.93 | 887.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 12:45:00 | 887.60 | 895.69 | 887.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:30:00 | 887.40 | 895.61 | 887.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 875.10 | 895.15 | 887.76 | SL hit (close<static) qty=1.00 sl=878.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 875.10 | 895.15 | 887.76 | SL hit (close<static) qty=1.00 sl=878.95 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 887.70 | 893.21 | 887.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:45:00 | 889.00 | 893.22 | 887.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 886.10 | 893.15 | 887.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 886.10 | 893.15 | 887.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 883.85 | 893.06 | 887.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 884.90 | 893.06 | 887.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 882.45 | 892.95 | 887.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 883.00 | 892.95 | 887.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 879.60 | 892.72 | 887.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 879.60 | 892.72 | 887.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-19 15:15:00 | 878.00 | 892.57 | 887.10 | SL hit (close<static) qty=1.00 sl=878.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 15:15:00 | 878.00 | 892.57 | 887.10 | SL hit (close<static) qty=1.00 sl=878.95 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 889.70 | 891.17 | 886.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 889.60 | 891.17 | 886.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 893.50 | 891.28 | 886.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 888.10 | 891.28 | 886.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 885.30 | 891.31 | 887.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 885.30 | 891.31 | 887.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 882.45 | 891.22 | 887.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 882.50 | 891.22 | 887.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 889.40 | 891.20 | 887.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 889.40 | 891.20 | 887.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 887.10 | 891.16 | 887.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 868.00 | 891.16 | 887.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 876.10 | 891.01 | 886.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:00:00 | 880.30 | 890.75 | 886.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:45:00 | 880.80 | 890.65 | 886.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:00:00 | 880.75 | 890.55 | 886.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 860.00 | 888.72 | 886.07 | SL hit (close<static) qty=1.00 sl=867.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 860.00 | 888.72 | 886.07 | SL hit (close<static) qty=1.00 sl=867.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 860.00 | 888.72 | 886.07 | SL hit (close<static) qty=1.00 sl=867.10 alert=retest2 |
| CROSSOVER_SKIP | 2025-09-02 09:15:00 | 831.75 | 883.52 | 883.57 | min_gap filter: gap=0.006% < 0.030% |
| TREND_RESET | 2025-09-02 09:15:00 | 831.75 | 883.52 | 883.57 | EMA inversion without crossover edge (EMA200=883.52 EMA400=883.57) — end cycle |
| CROSSOVER_SKIP | 2025-09-19 09:15:00 | 894.90 | 881.17 | 881.11 | min_gap filter: gap=0.007% < 0.030% |
| CROSSOVER_SKIP | 2025-09-26 14:15:00 | 857.40 | 881.10 | 881.17 | min_gap filter: gap=0.009% < 0.030% |
| CROSSOVER_SKIP | 2025-11-03 09:15:00 | 967.70 | 872.46 | 872.34 | min_gap filter: gap=0.012% < 0.030% |
| CROSSOVER_SKIP | 2025-12-02 15:15:00 | 845.55 | 880.35 | 880.45 | min_gap filter: gap=0.012% < 0.030% |
| CROSSOVER_SKIP | 2026-03-13 11:15:00 | 808.35 | 805.07 | 805.06 | min_gap filter: gap=0.002% < 0.030% |
| CROSSOVER_SKIP | 2026-03-13 13:15:00 | 796.25 | 804.96 | 805.00 | min_gap filter: gap=0.005% < 0.030% |
| CROSSOVER_SKIP | 2026-03-18 11:15:00 | 818.00 | 805.00 | 805.00 | min_gap filter: gap=0.000% < 0.030% |
| CROSSOVER_SKIP | 2026-03-23 10:15:00 | 767.20 | 804.69 | 804.87 | min_gap filter: gap=0.023% < 0.030% |
| CROSSOVER_SKIP | 2026-03-27 10:15:00 | 816.50 | 804.90 | 804.89 | min_gap filter: gap=0.001% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-13 12:45:00 | 887.60 | 2025-08-14 09:15:00 | 875.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-08-13 13:30:00 | 887.40 | 2025-08-14 09:15:00 | 875.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-08-19 09:15:00 | 887.70 | 2025-08-19 15:15:00 | 878.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-19 09:45:00 | 889.00 | 2025-08-19 15:15:00 | 878.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-26 12:00:00 | 880.30 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-26 12:45:00 | 880.80 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-08-26 14:00:00 | 880.75 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.36% |
