# Chennai Petroleum Corporation Ltd. (CHENNPETRO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1079.90
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
| ALERT2_SKIP | 1 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 3 |
| TARGET_HIT | 6 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 5 / 10 / 3
- **Avg / median % per leg:** 2.45% / -0.06%
- **Sum % (uncompounded):** 44.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 5 | 2 | 0 | 6.68% | 46.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 5 | 71.4% | 5 | 2 | 0 | 6.68% | 46.8% |
| SELL (all) | 11 | 3 | 27.3% | 0 | 8 | 3 | -0.24% | -2.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 3 | 27.3% | 0 | 8 | 3 | -0.24% | -2.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 8 | 44.4% | 5 | 10 | 3 | 2.45% | 44.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 14:15:00 | 656.25 | 676.91 | 676.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 648.00 | 671.97 | 674.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 668.55 | 668.51 | 672.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:45:00 | 669.85 | 668.51 | 672.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 683.00 | 668.65 | 672.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 681.50 | 668.65 | 672.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 684.50 | 668.81 | 672.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 684.25 | 668.81 | 672.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 700.00 | 675.68 | 675.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 702.65 | 675.95 | 675.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 744.75 | 757.21 | 731.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 10:00:00 | 744.75 | 757.21 | 731.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 724.50 | 755.91 | 731.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 724.50 | 755.91 | 731.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 723.05 | 755.58 | 731.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 723.05 | 755.58 | 731.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 752.50 | 753.76 | 731.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 733.90 | 753.76 | 731.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 901.55 | 939.80 | 886.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 885.40 | 939.80 | 886.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 899.20 | 933.53 | 894.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 899.20 | 933.53 | 894.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 890.75 | 932.03 | 894.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 892.95 | 932.03 | 894.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 889.85 | 931.61 | 894.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 888.75 | 931.61 | 894.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 895.00 | 930.31 | 894.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 894.50 | 930.31 | 894.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 894.90 | 929.96 | 894.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 894.75 | 929.96 | 894.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 894.80 | 929.61 | 894.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 894.00 | 929.61 | 894.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 891.40 | 929.23 | 894.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 891.40 | 929.23 | 894.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 886.00 | 928.80 | 894.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 890.80 | 928.80 | 894.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 886.05 | 927.92 | 894.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 892.95 | 926.80 | 894.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:00:00 | 893.90 | 926.47 | 894.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 878.65 | 925.27 | 894.46 | SL hit (close<static) qty=1.00 sl=879.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 878.65 | 925.27 | 894.46 | SL hit (close<static) qty=1.00 sl=879.60 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 810.40 | 873.91 | 873.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 801.50 | 871.41 | 872.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 860.10 | 854.11 | 863.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 10:15:00 | 860.10 | 854.11 | 863.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 860.10 | 854.11 | 863.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 860.10 | 854.11 | 863.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 868.50 | 854.26 | 863.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 859.15 | 854.29 | 863.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 885.25 | 854.93 | 863.32 | SL hit (close>static) qty=1.00 sl=878.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 852.70 | 854.91 | 863.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 858.00 | 855.22 | 863.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 854.10 | 855.24 | 863.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 815.10 | 854.17 | 862.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 811.39 | 853.80 | 862.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 810.07 | 852.95 | 861.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 858.50 | 850.38 | 859.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 858.50 | 850.38 | 859.86 | SL hit (close>ema200) qty=0.50 sl=850.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 858.50 | 850.38 | 859.86 | SL hit (close>ema200) qty=0.50 sl=850.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 858.50 | 850.38 | 859.86 | SL hit (close>ema200) qty=0.50 sl=850.38 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 859.85 | 850.38 | 859.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 866.25 | 848.74 | 858.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 867.35 | 848.74 | 858.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 855.00 | 848.80 | 858.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 861.00 | 848.80 | 858.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 844.00 | 848.54 | 857.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 863.85 | 848.54 | 857.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 860.00 | 848.65 | 857.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 860.00 | 848.65 | 857.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 852.90 | 848.69 | 857.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:00:00 | 847.20 | 848.68 | 857.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 846.10 | 848.74 | 857.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 871.50 | 849.05 | 857.82 | SL hit (close>static) qty=1.00 sl=860.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 871.50 | 849.05 | 857.82 | SL hit (close>static) qty=1.00 sl=860.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 822.75 | 851.56 | 858.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 11:00:00 | 846.30 | 850.91 | 857.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 865.40 | 850.74 | 857.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 865.40 | 850.74 | 857.73 | SL hit (close>static) qty=1.00 sl=860.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 865.40 | 850.74 | 857.73 | SL hit (close>static) qty=1.00 sl=860.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 865.40 | 850.74 | 857.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 868.80 | 850.92 | 857.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 852.15 | 850.92 | 857.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 875.00 | 850.54 | 857.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 875.00 | 850.54 | 857.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 882.10 | 850.86 | 857.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 882.10 | 850.86 | 857.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 919.85 | 863.40 | 863.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 933.40 | 867.27 | 865.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 872.25 | 874.60 | 869.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 872.25 | 874.60 | 869.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 902.35 | 918.13 | 898.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 948.00 | 918.42 | 898.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 922.55 | 920.00 | 899.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 13:15:00 | 921.00 | 920.07 | 900.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 922.05 | 920.09 | 900.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 927.95 | 920.04 | 900.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:15:00 | 945.95 | 920.14 | 900.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-17 10:15:00 | 1014.81 | 923.93 | 903.12 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-17 10:15:00 | 1013.10 | 923.93 | 903.12 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-17 10:15:00 | 1014.25 | 923.93 | 903.12 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-17 11:15:00 | 1042.80 | 925.07 | 903.80 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-17 11:15:00 | 1040.55 | 925.07 | 903.80 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-22 13:45:00 | 892.95 | 2025-12-23 10:15:00 | 878.65 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-12-22 15:00:00 | 893.90 | 2025-12-23 10:15:00 | 878.65 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-14 12:45:00 | 859.15 | 2026-01-16 09:15:00 | 885.25 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2026-01-16 11:00:00 | 852.70 | 2026-01-20 10:15:00 | 815.10 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2026-01-16 15:15:00 | 858.00 | 2026-01-20 11:15:00 | 811.39 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2026-01-19 10:00:00 | 854.10 | 2026-01-20 13:15:00 | 810.07 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-01-16 11:00:00 | 852.70 | 2026-01-22 09:15:00 | 858.50 | STOP_HIT | 0.50 | -0.68% |
| SELL | retest2 | 2026-01-16 15:15:00 | 858.00 | 2026-01-22 09:15:00 | 858.50 | STOP_HIT | 0.50 | -0.06% |
| SELL | retest2 | 2026-01-19 10:00:00 | 854.10 | 2026-01-22 09:15:00 | 858.50 | STOP_HIT | 0.50 | -0.52% |
| SELL | retest2 | 2026-01-28 13:00:00 | 847.20 | 2026-01-29 10:15:00 | 871.50 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-01-29 09:15:00 | 846.10 | 2026-01-29 10:15:00 | 871.50 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 822.75 | 2026-02-02 14:15:00 | 865.40 | STOP_HIT | 1.00 | -5.18% |
| SELL | retest2 | 2026-02-02 11:00:00 | 846.30 | 2026-02-02 14:15:00 | 865.40 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-12 10:45:00 | 948.00 | 2026-03-17 10:15:00 | 1014.81 | TARGET_HIT | 1.00 | 7.05% |
| BUY | retest2 | 2026-03-13 10:45:00 | 922.55 | 2026-03-17 10:15:00 | 1013.10 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2026-03-13 13:15:00 | 921.00 | 2026-03-17 10:15:00 | 1014.25 | TARGET_HIT | 1.00 | 10.13% |
| BUY | retest2 | 2026-03-13 14:00:00 | 922.05 | 2026-03-17 11:15:00 | 1042.80 | TARGET_HIT | 1.00 | 13.10% |
| BUY | retest2 | 2026-03-16 11:15:00 | 945.95 | 2026-03-17 11:15:00 | 1040.55 | TARGET_HIT | 1.00 | 10.00% |
