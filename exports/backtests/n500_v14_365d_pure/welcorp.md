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
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 23
- **Target hits / Stop hits / Partials:** 0 / 25 / 4
- **Avg / median % per leg:** -0.91% / -1.24%
- **Sum % (uncompounded):** -26.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.74% | -12.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.74% | -12.1% |
| SELL (all) | 22 | 6 | 27.3% | 0 | 18 | 4 | -0.65% | -14.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 0 | 18 | 4 | -0.65% | -14.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 6 | 20.7% | 0 | 25 | 4 | -0.91% | -26.5% |

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

### Cycle 2 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 831.75 | 883.52 | 883.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 828.40 | 882.97 | 883.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 873.95 | 872.86 | 877.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 873.95 | 872.86 | 877.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 870.25 | 872.83 | 877.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 869.65 | 872.84 | 877.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 881.20 | 872.92 | 877.66 | SL hit (close>static) qty=1.00 sl=880.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 869.95 | 874.03 | 877.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:30:00 | 869.40 | 873.97 | 877.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 880.45 | 874.01 | 877.72 | SL hit (close>static) qty=1.00 sl=880.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 880.45 | 874.01 | 877.72 | SL hit (close>static) qty=1.00 sl=880.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 894.90 | 881.17 | 881.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 905.95 | 881.74 | 881.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 13:15:00 | 882.10 | 882.57 | 881.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 882.10 | 882.57 | 881.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 885.30 | 882.61 | 881.88 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 857.40 | 881.10 | 881.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 845.15 | 879.54 | 880.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 876.80 | 873.21 | 876.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 879.40 | 873.21 | 876.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 876.00 | 873.24 | 876.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:15:00 | 872.00 | 873.24 | 876.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 15:00:00 | 871.40 | 873.18 | 876.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:15:00 | 828.40 | 869.07 | 874.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:15:00 | 827.83 | 869.07 | 874.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 872.70 | 853.84 | 864.12 | SL hit (close>ema200) qty=0.50 sl=853.84 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 872.70 | 853.84 | 864.12 | SL hit (close>ema200) qty=0.50 sl=853.84 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:00:00 | 872.70 | 853.84 | 864.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 874.20 | 854.02 | 864.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 907.50 | 855.43 | 864.57 | SL hit (close>static) qty=1.00 sl=881.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 907.50 | 855.43 | 864.57 | SL hit (close>static) qty=1.00 sl=881.95 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 967.70 | 872.46 | 872.34 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 845.55 | 880.35 | 880.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 837.45 | 879.92 | 880.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 829.50 | 826.09 | 845.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 829.50 | 826.09 | 845.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 799.30 | 765.71 | 794.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 777.30 | 795.29 | 802.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 780.15 | 795.14 | 802.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:00:00 | 780.70 | 794.79 | 802.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 778.65 | 794.31 | 801.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | SL hit (close>static) qty=1.00 sl=810.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | SL hit (close>static) qty=1.00 sl=810.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | SL hit (close>static) qty=1.00 sl=810.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 820.20 | 792.93 | 800.25 | SL hit (close>static) qty=1.00 sl=810.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 827.15 | 792.93 | 800.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 817.35 | 797.06 | 801.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 817.35 | 797.06 | 801.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 803.00 | 797.92 | 802.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 809.70 | 797.92 | 802.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 822.45 | 798.16 | 802.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 822.45 | 798.16 | 802.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 816.45 | 798.34 | 802.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 815.45 | 798.34 | 802.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 814.05 | 799.79 | 802.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 774.68 | 800.13 | 802.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 773.35 | 800.13 | 802.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 803.00 | 798.90 | 802.12 | SL hit (close>ema200) qty=0.50 sl=798.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 803.00 | 798.90 | 802.12 | SL hit (close>ema200) qty=0.50 sl=798.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 11:00:00 | 807.80 | 805.04 | 805.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 808.35 | 805.07 | 805.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 808.35 | 805.07 | 805.06 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 796.25 | 804.96 | 805.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 790.05 | 804.71 | 804.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 806.70 | 804.08 | 804.55 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 818.00 | 805.00 | 805.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 824.00 | 805.31 | 805.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 794.50 | 805.78 | 805.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 794.50 | 805.78 | 805.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 795.50 | 805.68 | 805.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 795.50 | 805.68 | 805.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 806.90 | 805.63 | 805.33 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 767.20 | 804.69 | 804.87 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 816.50 | 804.90 | 804.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 828.10 | 805.68 | 805.28 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 15:15:00 | 778.05 | 2025-05-19 09:15:00 | 793.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-20 12:30:00 | 780.10 | 2025-05-27 15:15:00 | 786.75 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-05-27 10:00:00 | 780.60 | 2025-05-27 15:15:00 | 786.75 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-05-27 12:15:00 | 780.85 | 2025-05-27 15:15:00 | 786.75 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-08-13 12:45:00 | 887.60 | 2025-08-14 09:15:00 | 875.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-08-13 13:30:00 | 887.40 | 2025-08-14 09:15:00 | 875.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-08-19 09:15:00 | 887.70 | 2025-08-19 15:15:00 | 878.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-19 09:45:00 | 889.00 | 2025-08-19 15:15:00 | 878.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-26 12:00:00 | 880.30 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-26 12:45:00 | 880.80 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-08-26 14:00:00 | 880.75 | 2025-08-29 09:15:00 | 860.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-09-09 11:30:00 | 869.65 | 2025-09-09 12:15:00 | 881.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-11 11:30:00 | 869.95 | 2025-09-12 15:15:00 | 880.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-11 12:30:00 | 869.40 | 2025-09-12 15:15:00 | 880.45 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-07 11:15:00 | 872.00 | 2025-10-13 09:15:00 | 828.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 15:00:00 | 871.40 | 2025-10-13 09:15:00 | 827.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 11:15:00 | 872.00 | 2025-10-27 09:15:00 | 872.70 | STOP_HIT | 0.50 | -0.08% |
| SELL | retest2 | 2025-10-07 15:00:00 | 871.40 | 2025-10-27 09:15:00 | 872.70 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-10-27 10:00:00 | 872.70 | 2025-10-28 09:15:00 | 907.50 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-10-27 10:45:00 | 874.20 | 2025-10-28 09:15:00 | 907.50 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2026-02-20 09:15:00 | 777.30 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2026-02-20 10:00:00 | 780.15 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2026-02-20 13:00:00 | 780.70 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2026-02-23 10:30:00 | 778.65 | 2026-02-26 09:15:00 | 820.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2026-03-05 11:15:00 | 815.45 | 2026-03-09 09:15:00 | 774.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 814.05 | 2026-03-09 09:15:00 | 773.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:15:00 | 815.45 | 2026-03-10 12:15:00 | 803.00 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2026-03-06 10:45:00 | 814.05 | 2026-03-10 12:15:00 | 803.00 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2026-03-13 11:00:00 | 807.80 | 2026-03-13 11:15:00 | 808.35 | STOP_HIT | 1.00 | -0.07% |
