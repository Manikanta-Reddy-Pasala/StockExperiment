# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1088.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 8 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 0 / 12 / 2
- **Avg / median % per leg:** -2.34% / -1.64%
- **Sum % (uncompounded):** -32.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 4 | 28.6% | 0 | 12 | 2 | -2.34% | -32.7% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -9.34% | -37.4% |
| SELL @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 0 | 8 | 2 | 0.46% | 4.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -9.34% | -37.4% |
| retest2 (combined) | 10 | 4 | 40.0% | 0 | 8 | 2 | 0.46% | 4.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 15:15:00)

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
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 890.65 | 852.21 | 878.13 | SL hit (close>static) qty=1.00 sl=886.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 878.10 | 860.26 | 879.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:30:00 | 878.00 | 860.63 | 879.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 876.00 | 860.78 | 879.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:00:00 | 874.70 | 860.92 | 879.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 874.00 | 861.23 | 879.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 875.05 | 861.52 | 879.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 873.25 | 862.04 | 879.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:15:00 | 834.19 | 857.89 | 874.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:15:00 | 834.10 | 857.89 | 874.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 858.20 | 856.70 | 872.88 | SL hit (close>ema200) qty=0.50 sl=856.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 858.20 | 856.70 | 872.88 | SL hit (close>ema200) qty=0.50 sl=856.70 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 875.30 | 856.97 | 872.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 875.30 | 856.97 | 872.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 872.40 | 857.12 | 872.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 874.30 | 857.12 | 872.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 867.70 | 857.23 | 872.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 889.40 | 857.97 | 872.89 | SL hit (close>static) qty=1.00 sl=882.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 889.40 | 857.97 | 872.89 | SL hit (close>static) qty=1.00 sl=882.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 889.40 | 857.97 | 872.89 | SL hit (close>static) qty=1.00 sl=882.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 889.40 | 857.97 | 872.89 | SL hit (close>static) qty=1.00 sl=882.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-03 10:15:00)

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

### Cycle 3 — SELL (started 2026-01-30 09:15:00)

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
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 853.75 | 800.83 | 826.33 | SL hit (close>ema400) qty=1.00 sl=826.33 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 853.75 | 800.83 | 826.33 | SL hit (close>ema400) qty=1.00 sl=826.33 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 853.75 | 800.83 | 826.33 | SL hit (close>ema400) qty=1.00 sl=826.33 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 853.80 | 800.83 | 826.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 979.70 | 844.90 | 844.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 987.85 | 847.65 | 845.65 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
