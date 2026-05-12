# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1736.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 23 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 64 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 51
- **Target hits / Stop hits / Partials:** 5 / 59 / 6
- **Avg / median % per leg:** -0.08% / -1.35%
- **Sum % (uncompounded):** -5.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 5 | 18.5% | 5 | 22 | 0 | 0.05% | 1.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 5 | 18.5% | 5 | 22 | 0 | 0.05% | 1.2% |
| SELL (all) | 43 | 14 | 32.6% | 0 | 37 | 6 | -0.17% | -7.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 14 | 32.6% | 0 | 37 | 6 | -0.17% | -7.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 19 | 27.1% | 5 | 59 | 6 | -0.08% | -5.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 715.85 | 703.22 | 702.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 716.65 | 707.93 | 705.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 876.25 | 876.57 | 835.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 872.50 | 876.57 | 835.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 876.95 | 882.06 | 871.26 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 857.90 | 867.42 | 867.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 855.25 | 864.45 | 866.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 13:15:00 | 858.10 | 853.38 | 857.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 13:15:00 | 858.10 | 853.38 | 857.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 858.10 | 853.38 | 857.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 855.30 | 853.38 | 857.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 856.00 | 853.91 | 857.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 856.00 | 853.91 | 857.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 878.25 | 857.99 | 858.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 878.25 | 857.99 | 858.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 877.50 | 861.89 | 860.19 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 857.25 | 862.18 | 862.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 850.35 | 858.85 | 860.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 872.85 | 861.65 | 861.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 872.85 | 861.65 | 861.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 872.85 | 861.65 | 861.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 868.10 | 861.65 | 861.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 891.20 | 867.56 | 864.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 908.45 | 887.89 | 877.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 13:15:00 | 897.35 | 898.99 | 887.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:45:00 | 900.00 | 898.99 | 887.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 895.00 | 899.94 | 890.56 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 881.60 | 888.22 | 888.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 875.05 | 883.11 | 885.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 879.50 | 876.29 | 880.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 879.50 | 876.29 | 880.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 879.50 | 876.29 | 880.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 882.30 | 876.29 | 880.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 879.70 | 876.97 | 880.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 879.75 | 876.97 | 880.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 879.65 | 877.51 | 880.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 875.00 | 877.01 | 879.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 874.90 | 876.48 | 878.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:45:00 | 874.30 | 875.65 | 877.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 873.00 | 874.31 | 876.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 890.00 | 874.70 | 875.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 890.00 | 874.70 | 875.16 | SL hit (close>static) qty=1.00 sl=880.25 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 888.25 | 877.41 | 876.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 912.45 | 886.27 | 880.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 899.05 | 900.37 | 892.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 898.80 | 900.37 | 892.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 903.75 | 900.84 | 895.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 932.20 | 903.61 | 899.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 917.40 | 903.61 | 899.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:30:00 | 922.15 | 910.94 | 903.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 10:15:00 | 890.55 | 898.68 | 899.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 890.55 | 898.68 | 899.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 11:15:00 | 884.50 | 895.84 | 898.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 871.00 | 868.92 | 879.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 871.00 | 868.92 | 879.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 884.75 | 871.78 | 878.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 884.75 | 871.78 | 878.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 875.40 | 872.51 | 878.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 873.10 | 872.51 | 878.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 870.15 | 871.62 | 877.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 829.44 | 844.31 | 852.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 826.64 | 844.31 | 852.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 842.75 | 841.62 | 848.13 | SL hit (close>ema200) qty=0.50 sl=841.62 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 857.60 | 848.16 | 847.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 865.50 | 854.79 | 851.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 855.00 | 855.18 | 852.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:00:00 | 855.00 | 855.18 | 852.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 853.75 | 854.90 | 852.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 853.75 | 854.90 | 852.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 853.50 | 854.62 | 853.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 864.30 | 854.62 | 853.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:30:00 | 859.70 | 862.04 | 859.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:45:00 | 860.40 | 861.99 | 860.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 864.00 | 861.21 | 859.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 864.00 | 861.77 | 860.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 853.20 | 861.77 | 860.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 856.65 | 860.74 | 859.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 850.40 | 860.74 | 859.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 851.30 | 858.85 | 859.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 851.30 | 858.85 | 859.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 11:15:00 | 849.90 | 857.06 | 858.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 857.70 | 853.50 | 855.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 857.70 | 853.50 | 855.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 857.70 | 853.50 | 855.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 854.95 | 853.72 | 855.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 854.25 | 854.27 | 855.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 854.00 | 854.53 | 855.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 854.45 | 855.26 | 855.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 841.95 | 852.60 | 854.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 855.85 | 850.88 | 850.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 855.85 | 850.88 | 850.24 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 844.90 | 849.73 | 850.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 840.10 | 846.14 | 848.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 839.10 | 838.54 | 842.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 839.10 | 838.54 | 842.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 839.10 | 838.54 | 842.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 837.85 | 838.54 | 842.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 839.60 | 838.75 | 842.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 835.10 | 838.02 | 841.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 835.30 | 837.19 | 840.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 845.60 | 837.94 | 840.08 | SL hit (close>static) qty=1.00 sl=845.55 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 854.10 | 843.50 | 842.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 12:15:00 | 865.40 | 847.88 | 844.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 912.35 | 913.70 | 889.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:30:00 | 915.20 | 913.70 | 889.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 930.50 | 940.06 | 933.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 928.80 | 937.81 | 932.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 920.00 | 934.25 | 931.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 920.00 | 934.25 | 931.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 917.20 | 928.14 | 929.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 912.25 | 916.93 | 920.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 919.10 | 913.48 | 917.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 919.10 | 913.48 | 917.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 919.10 | 913.48 | 917.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 917.80 | 913.48 | 917.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 915.85 | 913.95 | 917.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 912.75 | 913.71 | 916.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 913.75 | 913.75 | 915.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 922.95 | 915.59 | 916.49 | SL hit (close>static) qty=1.00 sl=919.85 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 920.25 | 917.62 | 917.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 939.80 | 922.06 | 919.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 917.55 | 924.62 | 921.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 10:15:00 | 917.55 | 924.62 | 921.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 917.55 | 924.62 | 921.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 917.70 | 924.62 | 921.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 912.00 | 922.09 | 920.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 912.00 | 922.09 | 920.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 911.80 | 920.03 | 919.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 913.00 | 920.03 | 919.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 916.15 | 919.26 | 919.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 907.90 | 915.60 | 917.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 909.55 | 905.07 | 909.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 909.55 | 905.07 | 909.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 909.55 | 905.07 | 909.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 909.00 | 905.07 | 909.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 914.90 | 907.04 | 909.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 914.90 | 907.04 | 909.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 916.05 | 908.84 | 910.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 916.05 | 908.84 | 910.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 923.00 | 911.67 | 911.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 924.65 | 918.96 | 915.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 917.50 | 919.05 | 916.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 917.50 | 919.05 | 916.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 918.00 | 918.84 | 916.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 912.30 | 918.84 | 916.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 907.60 | 916.59 | 915.52 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 905.70 | 914.41 | 914.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 901.90 | 910.28 | 912.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 886.25 | 879.62 | 888.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 886.25 | 879.62 | 888.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 886.25 | 879.62 | 888.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 886.25 | 879.62 | 888.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 882.00 | 880.10 | 887.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 890.20 | 880.10 | 887.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 878.50 | 879.78 | 886.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 871.45 | 878.47 | 885.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 929.30 | 881.52 | 882.73 | SL hit (close>static) qty=1.00 sl=890.70 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 10:15:00 | 905.15 | 886.25 | 884.77 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 894.40 | 902.75 | 903.27 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 921.55 | 902.40 | 901.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 922.95 | 914.14 | 909.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 958.45 | 959.20 | 948.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 958.45 | 959.20 | 948.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 947.05 | 955.34 | 951.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 947.05 | 955.34 | 951.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 951.00 | 954.47 | 951.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 958.40 | 956.66 | 952.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 955.80 | 955.33 | 952.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 955.05 | 954.79 | 953.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 957.55 | 954.27 | 953.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 980.15 | 959.45 | 955.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 925.30 | 954.41 | 957.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 925.30 | 954.41 | 957.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 919.50 | 942.83 | 951.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 908.85 | 907.86 | 922.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 908.85 | 907.86 | 922.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 905.95 | 901.01 | 909.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 905.95 | 901.01 | 909.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 916.00 | 904.01 | 909.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 915.70 | 904.01 | 909.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 915.50 | 906.31 | 910.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 921.20 | 906.31 | 910.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 928.05 | 914.44 | 913.51 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 890.05 | 911.88 | 913.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 865.05 | 879.39 | 889.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 14:15:00 | 865.95 | 864.43 | 872.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 865.95 | 864.43 | 872.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 869.70 | 865.49 | 871.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 872.60 | 865.49 | 871.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 872.00 | 866.79 | 871.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 876.30 | 866.79 | 871.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 867.55 | 866.94 | 871.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 864.85 | 866.54 | 870.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 876.55 | 867.66 | 869.73 | SL hit (close>static) qty=1.00 sl=872.30 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 892.55 | 872.64 | 871.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 901.80 | 891.50 | 888.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 941.50 | 942.61 | 930.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:45:00 | 938.35 | 942.61 | 930.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 963.50 | 963.42 | 956.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 960.90 | 963.42 | 956.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 957.65 | 961.39 | 957.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 958.45 | 961.39 | 957.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 952.45 | 959.61 | 957.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 952.45 | 959.61 | 957.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 957.10 | 959.10 | 957.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 953.60 | 959.10 | 957.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 966.55 | 960.59 | 957.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 971.00 | 962.65 | 959.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 952.50 | 961.97 | 962.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 952.50 | 961.97 | 962.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 942.95 | 954.57 | 958.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 927.45 | 923.49 | 934.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 927.45 | 923.49 | 934.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 920.00 | 922.79 | 932.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 921.60 | 922.79 | 932.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 923.75 | 917.31 | 923.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 930.50 | 917.31 | 923.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 927.80 | 919.41 | 924.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 910.60 | 920.53 | 922.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 893.70 | 886.71 | 886.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 893.70 | 886.71 | 886.41 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 881.95 | 886.64 | 886.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 878.65 | 885.04 | 885.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 885.05 | 883.81 | 885.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 885.05 | 883.81 | 885.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 885.05 | 883.81 | 885.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 886.30 | 883.81 | 885.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 884.00 | 883.85 | 884.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 885.65 | 883.85 | 884.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 884.20 | 883.92 | 884.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 883.75 | 883.92 | 884.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 874.75 | 882.09 | 883.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:30:00 | 872.00 | 880.18 | 882.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 872.55 | 880.18 | 882.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 872.30 | 877.01 | 880.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:00:00 | 867.65 | 875.14 | 879.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 877.35 | 874.23 | 877.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 877.35 | 874.23 | 877.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 874.05 | 874.20 | 877.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 890.00 | 878.40 | 878.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 890.00 | 878.40 | 878.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 896.05 | 881.93 | 879.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 884.10 | 884.66 | 881.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:00:00 | 884.10 | 884.66 | 881.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 878.25 | 883.38 | 881.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 878.25 | 883.38 | 881.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 880.10 | 882.73 | 881.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 889.00 | 883.76 | 881.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 886.00 | 886.77 | 884.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:00:00 | 887.50 | 887.27 | 885.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 880.80 | 883.35 | 883.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 880.80 | 883.35 | 883.63 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 895.80 | 884.92 | 884.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 900.95 | 889.78 | 886.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 894.00 | 894.94 | 890.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 894.00 | 894.94 | 890.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 890.20 | 893.99 | 890.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 903.85 | 893.99 | 890.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 898.40 | 898.58 | 895.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:00:00 | 898.60 | 898.59 | 895.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 988.24 | 935.04 | 916.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 993.50 | 1009.05 | 1010.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 981.80 | 1003.60 | 1007.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 946.90 | 945.99 | 956.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:45:00 | 947.00 | 945.99 | 956.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 33 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1076.40 | 972.23 | 966.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 1088.15 | 1012.73 | 987.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1162.50 | 1175.27 | 1160.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1162.50 | 1175.27 | 1160.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1162.50 | 1175.27 | 1160.30 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 1140.15 | 1157.38 | 1158.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 1134.00 | 1150.00 | 1155.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 1146.00 | 1141.13 | 1148.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:45:00 | 1144.25 | 1141.13 | 1148.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1151.90 | 1143.28 | 1148.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 1151.90 | 1143.28 | 1148.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 1158.60 | 1146.35 | 1149.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 1155.65 | 1146.35 | 1149.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1156.00 | 1149.49 | 1150.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 1152.00 | 1149.49 | 1150.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1162.45 | 1153.14 | 1152.13 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1145.95 | 1151.13 | 1151.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 1139.30 | 1146.94 | 1149.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 1147.70 | 1143.72 | 1147.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 1147.70 | 1143.72 | 1147.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1147.70 | 1143.72 | 1147.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 1147.70 | 1143.72 | 1147.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1139.85 | 1142.95 | 1146.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 1133.25 | 1140.35 | 1144.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 1134.40 | 1132.71 | 1137.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 1130.60 | 1132.19 | 1136.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1115.20 | 1119.05 | 1124.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1102.40 | 1115.72 | 1122.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:45:00 | 1095.00 | 1110.98 | 1119.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 1076.59 | 1103.64 | 1115.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 1077.68 | 1103.64 | 1115.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 1074.07 | 1103.64 | 1115.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1106.60 | 1097.38 | 1107.97 | SL hit (close>ema200) qty=0.50 sl=1097.38 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1162.50 | 1123.29 | 1118.14 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1101.70 | 1127.99 | 1128.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1095.20 | 1121.44 | 1125.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1116.30 | 1111.44 | 1118.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 1116.30 | 1111.44 | 1118.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1119.50 | 1113.05 | 1118.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 1123.40 | 1113.05 | 1118.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1117.80 | 1114.00 | 1118.68 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1137.00 | 1122.28 | 1121.33 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 1114.40 | 1121.46 | 1121.55 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1131.30 | 1121.53 | 1120.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1138.50 | 1126.91 | 1123.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 1137.20 | 1139.89 | 1133.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 1137.20 | 1139.89 | 1133.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1136.10 | 1139.13 | 1133.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1152.70 | 1139.13 | 1133.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-17 09:15:00 | 1267.97 | 1211.08 | 1177.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1267.00 | 1280.74 | 1282.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 1260.00 | 1274.22 | 1279.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 1264.00 | 1261.78 | 1269.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 1264.00 | 1261.78 | 1269.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1264.00 | 1261.78 | 1269.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1264.00 | 1261.78 | 1269.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1266.70 | 1262.77 | 1269.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1266.70 | 1262.77 | 1269.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1270.00 | 1264.21 | 1269.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1260.60 | 1264.21 | 1269.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1233.10 | 1229.03 | 1228.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1233.10 | 1229.03 | 1228.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1244.00 | 1232.02 | 1229.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 1248.90 | 1252.21 | 1244.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:45:00 | 1246.50 | 1252.21 | 1244.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1249.40 | 1251.65 | 1244.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 1243.20 | 1251.65 | 1244.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1251.30 | 1252.04 | 1246.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1264.70 | 1252.04 | 1246.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1261.00 | 1252.49 | 1250.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1222.10 | 1246.41 | 1247.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 1222.10 | 1246.41 | 1247.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1200.40 | 1220.54 | 1231.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 13:15:00 | 1135.00 | 1132.60 | 1148.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:45:00 | 1137.00 | 1132.60 | 1148.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1150.00 | 1138.54 | 1148.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1145.80 | 1138.83 | 1148.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 1165.30 | 1149.43 | 1151.04 | SL hit (close>static) qty=1.00 sl=1160.10 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1166.00 | 1152.75 | 1152.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 1171.80 | 1158.76 | 1155.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 1171.10 | 1173.66 | 1166.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 1171.10 | 1173.66 | 1166.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1171.10 | 1173.66 | 1166.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1171.10 | 1173.66 | 1166.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1160.00 | 1170.93 | 1165.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1146.00 | 1170.93 | 1165.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1142.80 | 1165.30 | 1163.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1142.80 | 1165.30 | 1163.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1130.00 | 1158.24 | 1160.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1104.70 | 1133.81 | 1145.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1095.10 | 1094.78 | 1112.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1095.10 | 1094.78 | 1112.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1106.00 | 1097.52 | 1109.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1125.50 | 1097.52 | 1109.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1123.50 | 1102.72 | 1110.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1123.70 | 1102.72 | 1110.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1101.00 | 1102.37 | 1110.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1097.20 | 1102.92 | 1109.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1097.20 | 1107.15 | 1109.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 1094.10 | 1102.01 | 1105.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 1097.20 | 1100.88 | 1104.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1090.50 | 1093.64 | 1099.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1109.60 | 1093.64 | 1099.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1104.20 | 1095.75 | 1099.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1102.70 | 1095.75 | 1099.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1101.80 | 1096.96 | 1099.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1101.80 | 1096.96 | 1099.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1104.30 | 1098.43 | 1100.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1104.30 | 1098.43 | 1100.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1102.40 | 1099.22 | 1100.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 1113.90 | 1102.89 | 1101.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 1113.90 | 1102.89 | 1101.97 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1088.50 | 1100.99 | 1101.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 1084.20 | 1097.63 | 1099.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 11:15:00 | 1105.40 | 1099.19 | 1100.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 1105.40 | 1099.19 | 1100.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 1105.40 | 1099.19 | 1100.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:30:00 | 1105.60 | 1099.19 | 1100.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1110.00 | 1101.35 | 1101.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1121.50 | 1105.38 | 1103.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1149.00 | 1155.21 | 1137.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 1145.70 | 1155.21 | 1137.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1140.20 | 1152.21 | 1137.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1136.60 | 1152.21 | 1137.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1121.90 | 1146.15 | 1136.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1121.90 | 1146.15 | 1136.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1137.70 | 1144.46 | 1136.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1123.20 | 1144.46 | 1136.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1128.50 | 1140.57 | 1136.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 1114.30 | 1140.57 | 1136.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1126.60 | 1137.77 | 1135.56 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1105.20 | 1131.26 | 1132.80 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1154.40 | 1135.58 | 1133.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1164.30 | 1141.33 | 1136.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 13:15:00 | 1198.00 | 1204.01 | 1188.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:00:00 | 1198.00 | 1204.01 | 1188.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1176.20 | 1196.59 | 1188.56 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1160.30 | 1182.71 | 1183.68 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1203.30 | 1183.60 | 1183.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 1224.60 | 1210.97 | 1199.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 14:15:00 | 1336.80 | 1350.07 | 1308.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 1336.80 | 1350.07 | 1308.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1415.10 | 1416.10 | 1397.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 1406.80 | 1416.10 | 1397.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1421.00 | 1417.08 | 1400.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 1432.60 | 1418.73 | 1402.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1432.50 | 1420.02 | 1409.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:30:00 | 1428.20 | 1420.78 | 1410.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 1401.00 | 1412.18 | 1413.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 1401.00 | 1412.18 | 1413.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1397.00 | 1409.14 | 1411.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 1391.50 | 1389.14 | 1396.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 1391.50 | 1389.14 | 1396.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1391.50 | 1389.14 | 1396.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 1391.50 | 1389.14 | 1396.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1384.00 | 1388.11 | 1395.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1412.20 | 1388.11 | 1395.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1420.60 | 1394.61 | 1398.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1422.00 | 1394.61 | 1398.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1412.50 | 1398.19 | 1399.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1410.60 | 1398.19 | 1399.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1409.60 | 1401.60 | 1400.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1409.60 | 1401.60 | 1400.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1419.60 | 1405.42 | 1402.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1403.00 | 1406.73 | 1403.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 1403.00 | 1406.73 | 1403.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1403.00 | 1406.73 | 1403.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 1403.00 | 1406.73 | 1403.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1392.80 | 1403.94 | 1402.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 1394.40 | 1403.94 | 1402.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1391.00 | 1401.35 | 1401.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1371.00 | 1393.46 | 1397.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 1386.90 | 1385.22 | 1392.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 1387.20 | 1386.27 | 1391.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1387.20 | 1386.27 | 1391.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1387.20 | 1386.27 | 1391.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1395.00 | 1388.02 | 1391.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1358.90 | 1388.02 | 1391.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1290.95 | 1384.29 | 1389.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 1407.10 | 1373.31 | 1380.46 | SL hit (close>ema200) qty=0.50 sl=1373.31 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1420.90 | 1382.92 | 1382.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 1439.30 | 1394.05 | 1387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1486.40 | 1496.98 | 1467.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1486.40 | 1496.98 | 1467.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1486.40 | 1496.98 | 1467.59 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 1428.70 | 1460.43 | 1460.97 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1467.80 | 1449.73 | 1448.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 15:15:00 | 1479.00 | 1460.35 | 1453.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 11:15:00 | 1447.40 | 1460.72 | 1456.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 1447.40 | 1460.72 | 1456.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 1447.40 | 1460.72 | 1456.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 1447.40 | 1460.72 | 1456.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 1433.70 | 1455.31 | 1454.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:45:00 | 1434.70 | 1455.31 | 1454.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 1442.40 | 1451.62 | 1452.50 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 09:15:00 | 1473.40 | 1455.97 | 1454.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 11:15:00 | 1501.80 | 1469.00 | 1460.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 14:15:00 | 1473.20 | 1474.04 | 1465.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 1473.20 | 1474.04 | 1465.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1461.00 | 1471.43 | 1465.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1456.00 | 1471.43 | 1465.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1450.90 | 1467.32 | 1463.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 1450.20 | 1467.32 | 1463.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1451.90 | 1464.24 | 1462.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1451.90 | 1464.24 | 1462.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 1448.10 | 1461.01 | 1461.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 14:15:00 | 1435.20 | 1449.08 | 1453.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1318.50 | 1302.15 | 1336.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1318.50 | 1302.15 | 1336.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1390.00 | 1329.85 | 1339.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1390.00 | 1329.85 | 1339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1396.40 | 1343.16 | 1344.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 1395.00 | 1343.16 | 1344.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1393.70 | 1353.27 | 1349.23 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1330.50 | 1355.25 | 1357.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1319.40 | 1348.08 | 1353.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1398.90 | 1341.86 | 1344.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1398.90 | 1341.86 | 1344.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1398.90 | 1341.86 | 1344.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1406.40 | 1341.86 | 1344.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1382.70 | 1350.03 | 1347.62 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1334.00 | 1349.63 | 1350.18 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1355.20 | 1351.19 | 1350.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1367.70 | 1354.49 | 1352.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1393.00 | 1404.75 | 1388.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1393.00 | 1404.75 | 1388.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1393.00 | 1404.75 | 1388.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1424.50 | 1398.05 | 1391.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1566.95 | 1499.82 | 1479.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1611.90 | 1637.65 | 1638.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1592.50 | 1617.01 | 1626.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1675.70 | 1612.97 | 1617.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1675.70 | 1612.97 | 1617.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1675.70 | 1612.97 | 1617.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1675.70 | 1612.97 | 1617.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1685.00 | 1627.37 | 1623.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1720.00 | 1672.31 | 1650.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1700.00 | 1715.43 | 1696.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 1700.00 | 1715.43 | 1696.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1700.00 | 1709.70 | 1696.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1702.20 | 1709.70 | 1696.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1686.30 | 1705.02 | 1695.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1686.30 | 1705.02 | 1695.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1700.00 | 1704.02 | 1696.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 1709.10 | 1700.54 | 1696.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 1708.50 | 1699.85 | 1696.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1684.30 | 1723.49 | 1720.33 | SL hit (close<static) qty=1.00 sl=1685.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 1692.40 | 1717.28 | 1717.79 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1752.50 | 1717.61 | 1716.28 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:30:00 | 715.20 | 2025-05-13 09:15:00 | 715.85 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-05-12 11:00:00 | 714.95 | 2025-05-13 09:15:00 | 715.85 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-06-04 15:00:00 | 875.00 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-05 11:00:00 | 874.90 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-06 09:45:00 | 874.30 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-06-06 10:30:00 | 873.00 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-12 09:30:00 | 932.20 | 2025-06-13 10:15:00 | 890.55 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-06-12 10:00:00 | 917.40 | 2025-06-13 10:15:00 | 890.55 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-06-12 12:30:00 | 922.15 | 2025-06-13 10:15:00 | 890.55 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-06-17 11:15:00 | 873.10 | 2025-06-20 09:15:00 | 829.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 870.15 | 2025-06-20 09:15:00 | 826.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 873.10 | 2025-06-20 14:15:00 | 842.75 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-06-17 11:45:00 | 870.15 | 2025-06-20 14:15:00 | 842.75 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2025-06-26 09:15:00 | 864.30 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-27 12:30:00 | 859.70 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-27 13:45:00 | 860.40 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-27 15:15:00 | 864.00 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-07-01 10:45:00 | 854.95 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-01 13:15:00 | 854.25 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-07-01 13:45:00 | 854.00 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-07-02 09:15:00 | 854.45 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-07-09 12:00:00 | 835.10 | 2025-07-10 09:15:00 | 845.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-09 13:30:00 | 835.30 | 2025-07-10 09:15:00 | 845.60 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-23 13:00:00 | 912.75 | 2025-07-24 09:15:00 | 922.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-24 09:15:00 | 913.75 | 2025-07-24 09:15:00 | 922.95 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-05 10:45:00 | 871.45 | 2025-08-06 09:15:00 | 929.30 | STOP_HIT | 1.00 | -6.64% |
| BUY | retest2 | 2025-08-22 09:45:00 | 958.40 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-22 11:30:00 | 955.80 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-08-22 13:30:00 | 955.05 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-08-25 09:30:00 | 957.55 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-09-09 14:00:00 | 864.85 | 2025-09-10 09:15:00 | 876.55 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-24 10:45:00 | 971.00 | 2025-09-25 12:15:00 | 952.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-10-03 10:30:00 | 910.60 | 2025-10-10 13:15:00 | 893.70 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2025-10-14 11:30:00 | 872.00 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-10-14 12:00:00 | 872.55 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-14 13:45:00 | 872.30 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-10-14 15:00:00 | 867.65 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-10-17 09:30:00 | 889.00 | 2025-10-20 13:15:00 | 880.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-17 13:45:00 | 886.00 | 2025-10-20 13:15:00 | 880.80 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-20 10:00:00 | 887.50 | 2025-10-20 13:15:00 | 880.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-24 09:15:00 | 903.85 | 2025-10-28 09:15:00 | 988.24 | TARGET_HIT | 1.00 | 9.34% |
| BUY | retest2 | 2025-10-27 09:15:00 | 898.40 | 2025-10-28 09:15:00 | 988.46 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2025-10-27 10:00:00 | 898.60 | 2025-10-28 10:15:00 | 994.24 | TARGET_HIT | 1.00 | 10.64% |
| SELL | retest2 | 2025-11-28 10:45:00 | 1133.25 | 2025-12-03 12:15:00 | 1076.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 1134.40 | 2025-12-03 12:15:00 | 1077.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1130.60 | 2025-12-03 12:15:00 | 1074.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 10:45:00 | 1133.25 | 2025-12-04 09:15:00 | 1106.60 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2025-12-01 10:45:00 | 1134.40 | 2025-12-04 09:15:00 | 1106.60 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1130.60 | 2025-12-04 09:15:00 | 1106.60 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-12-03 09:30:00 | 1115.20 | 2025-12-04 11:15:00 | 1157.80 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-12-03 11:45:00 | 1095.00 | 2025-12-04 11:15:00 | 1157.80 | STOP_HIT | 1.00 | -5.74% |
| BUY | retest2 | 2025-12-16 09:15:00 | 1152.70 | 2025-12-17 09:15:00 | 1267.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1260.60 | 2026-01-02 09:15:00 | 1233.10 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1264.70 | 2026-01-07 10:15:00 | 1222.10 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1261.00 | 2026-01-07 10:15:00 | 1222.10 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1145.80 | 2026-01-14 12:15:00 | 1165.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1097.20 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-23 13:15:00 | 1097.20 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-27 10:15:00 | 1094.10 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-01-27 12:15:00 | 1097.20 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-02-18 11:45:00 | 1432.60 | 2026-02-23 10:15:00 | 1401.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-02-19 09:30:00 | 1432.50 | 2026-02-23 10:15:00 | 1401.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-02-19 10:30:00 | 1428.20 | 2026-02-23 10:15:00 | 1401.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1410.60 | 2026-02-25 14:15:00 | 1409.60 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1358.90 | 2026-03-02 09:15:00 | 1290.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1358.90 | 2026-03-02 14:15:00 | 1407.10 | STOP_HIT | 0.50 | -3.55% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1362.90 | 2026-03-04 14:15:00 | 1420.90 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1424.50 | 2026-04-15 09:15:00 | 1566.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 13:30:00 | 1709.10 | 2026-05-06 09:15:00 | 1684.30 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-04-30 15:15:00 | 1708.50 | 2026-05-06 09:15:00 | 1684.30 | STOP_HIT | 1.00 | -1.42% |
