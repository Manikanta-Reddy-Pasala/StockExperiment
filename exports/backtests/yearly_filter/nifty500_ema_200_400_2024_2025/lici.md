# Life Insurance Corporation of India (LICI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 802.45
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
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 7 / 33
- **Target hits / Stop hits / Partials:** 2 / 36 / 2
- **Avg / median % per leg:** -0.41% / -0.91%
- **Sum % (uncompounded):** -16.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 3 | 15.0% | 2 | 18 | 0 | 0.24% | 4.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 3 | 15.0% | 2 | 18 | 0 | 0.24% | 4.7% |
| SELL (all) | 20 | 4 | 20.0% | 0 | 18 | 2 | -1.06% | -21.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 4 | 20.0% | 0 | 18 | 2 | -1.06% | -21.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 7 | 17.5% | 2 | 36 | 2 | -0.41% | -16.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 14:15:00 | 1022.00 | 1057.41 | 1057.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 15:15:00 | 1020.00 | 1057.04 | 1057.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 953.50 | 946.32 | 976.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:00:00 | 953.50 | 946.32 | 976.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 969.55 | 926.59 | 954.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 986.50 | 926.59 | 954.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 975.00 | 927.07 | 954.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 975.00 | 927.07 | 954.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 804.60 | 774.86 | 805.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:30:00 | 806.50 | 774.86 | 805.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 806.70 | 775.17 | 805.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 806.70 | 775.17 | 805.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 801.05 | 775.43 | 805.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:00:00 | 797.30 | 775.91 | 805.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 811.00 | 779.21 | 803.90 | SL hit (close>static) qty=1.00 sl=806.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 857.20 | 804.77 | 804.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 861.20 | 806.36 | 805.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 934.45 | 935.83 | 905.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 15:15:00 | 918.00 | 929.56 | 911.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 918.00 | 929.56 | 911.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 903.80 | 929.56 | 911.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 895.50 | 929.22 | 911.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 895.50 | 929.22 | 911.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 896.30 | 910.97 | 905.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 896.30 | 910.97 | 905.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 895.10 | 910.81 | 905.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 895.10 | 910.81 | 905.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 905.10 | 910.39 | 905.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 923.80 | 910.39 | 905.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 917.75 | 910.47 | 905.83 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 887.95 | 902.82 | 902.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 877.15 | 902.57 | 902.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 888.70 | 885.47 | 891.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 12:00:00 | 888.70 | 885.47 | 891.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 888.95 | 885.41 | 891.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 887.30 | 885.58 | 891.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 894.20 | 885.75 | 891.46 | SL hit (close>static) qty=1.00 sl=892.95 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 898.60 | 894.45 | 894.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 901.00 | 894.58 | 894.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 894.00 | 895.31 | 894.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 894.00 | 895.31 | 894.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 894.00 | 895.31 | 894.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 894.00 | 895.31 | 894.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 893.65 | 895.29 | 894.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 891.70 | 895.29 | 894.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 894.15 | 895.27 | 894.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 894.15 | 895.27 | 894.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 892.80 | 895.25 | 894.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 892.80 | 895.25 | 894.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 889.50 | 895.20 | 894.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 888.50 | 895.20 | 894.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 893.00 | 895.18 | 894.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:30:00 | 894.70 | 894.73 | 894.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 893.15 | 894.58 | 894.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 897.85 | 894.56 | 894.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 889.45 | 894.90 | 894.73 | SL hit (close<static) qty=1.00 sl=889.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 868.00 | 898.08 | 898.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 863.90 | 894.03 | 896.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 830.45 | 830.00 | 848.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:30:00 | 830.40 | 830.00 | 848.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 895.20 | 831.79 | 847.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 893.60 | 831.79 | 847.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 858.05 | 849.09 | 853.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 858.05 | 849.09 | 853.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 882.75 | 857.78 | 857.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 885.00 | 858.85 | 858.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 857.50 | 860.92 | 859.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 857.50 | 860.92 | 859.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 857.50 | 860.92 | 859.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 857.45 | 860.92 | 859.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 855.75 | 860.86 | 859.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 855.75 | 860.86 | 859.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 826.35 | 857.91 | 858.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 823.70 | 856.12 | 857.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 791.40 | 785.92 | 811.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 791.40 | 785.92 | 811.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 805.25 | 787.10 | 810.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:30:00 | 796.85 | 787.94 | 810.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 788.30 | 788.07 | 810.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 797.50 | 788.32 | 810.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 831.75 | 789.20 | 810.11 | SL hit (close>static) qty=1.00 sl=811.95 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 10:15:00 | 959.50 | 2024-05-22 09:15:00 | 1055.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:15:00 | 956.10 | 2024-06-04 14:15:00 | 933.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-06-06 09:15:00 | 983.45 | 2024-06-18 09:15:00 | 1081.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-25 10:00:00 | 797.30 | 2025-03-28 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-03-28 15:15:00 | 798.40 | 2025-04-02 13:15:00 | 809.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-04-01 10:15:00 | 798.15 | 2025-04-02 13:15:00 | 809.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-04-01 10:45:00 | 798.60 | 2025-04-02 13:15:00 | 809.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-04-04 09:45:00 | 798.40 | 2025-04-07 09:15:00 | 758.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 11:45:00 | 799.25 | 2025-04-07 09:15:00 | 759.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:45:00 | 798.40 | 2025-04-08 14:15:00 | 785.00 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2025-04-04 11:45:00 | 799.25 | 2025-04-08 14:15:00 | 785.00 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2025-04-17 12:00:00 | 799.60 | 2025-04-21 09:15:00 | 809.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-17 13:00:00 | 800.10 | 2025-04-21 09:15:00 | 809.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-25 11:45:00 | 789.55 | 2025-04-28 12:15:00 | 806.65 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-05-06 14:15:00 | 787.40 | 2025-05-12 09:15:00 | 808.80 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-05-08 13:15:00 | 789.15 | 2025-05-12 09:15:00 | 808.80 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-09-18 14:15:00 | 887.30 | 2025-09-19 09:15:00 | 894.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-09-25 09:30:00 | 886.20 | 2025-09-25 10:15:00 | 897.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-25 15:00:00 | 887.00 | 2025-09-29 14:15:00 | 895.65 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-20 10:30:00 | 894.70 | 2025-10-24 14:15:00 | 889.45 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-21 13:45:00 | 893.15 | 2025-10-24 14:15:00 | 889.45 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-10-23 09:15:00 | 897.85 | 2025-10-24 14:15:00 | 889.45 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-27 09:15:00 | 893.25 | 2025-10-31 13:15:00 | 893.40 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-10-27 14:15:00 | 898.85 | 2025-10-31 13:15:00 | 893.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-27 15:15:00 | 898.20 | 2025-10-31 13:15:00 | 893.40 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-31 10:45:00 | 898.20 | 2025-11-06 15:15:00 | 894.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-11-03 09:15:00 | 899.60 | 2025-11-06 15:15:00 | 894.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-11-06 13:15:00 | 898.80 | 2025-11-06 15:15:00 | 894.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-11-06 14:00:00 | 897.50 | 2025-11-24 10:15:00 | 894.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-11-06 14:30:00 | 897.70 | 2025-11-24 10:15:00 | 894.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-11-07 09:15:00 | 905.60 | 2025-11-24 10:15:00 | 894.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-11-07 12:15:00 | 901.95 | 2025-11-24 10:15:00 | 894.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-11 14:00:00 | 901.80 | 2025-11-24 10:15:00 | 894.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-11 15:00:00 | 902.20 | 2025-11-24 10:15:00 | 894.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-12 09:15:00 | 904.70 | 2025-11-24 11:15:00 | 890.80 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-14 13:15:00 | 906.40 | 2025-12-01 11:15:00 | 889.10 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-04-10 14:30:00 | 796.85 | 2026-04-15 09:15:00 | 831.75 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-04-13 09:15:00 | 788.30 | 2026-04-15 09:15:00 | 831.75 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2026-04-13 13:15:00 | 797.50 | 2026-04-15 09:15:00 | 831.75 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2026-04-30 15:00:00 | 797.30 | 2026-05-07 11:15:00 | 813.40 | STOP_HIT | 1.00 | -2.02% |
