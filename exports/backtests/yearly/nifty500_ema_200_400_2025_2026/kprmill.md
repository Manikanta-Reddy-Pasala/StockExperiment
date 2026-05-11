# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 955.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 6
- **Target hits / Stop hits / Partials:** 3 / 9 / 6
- **Avg / median % per leg:** 2.15% / 5.00%
- **Sum % (uncompounded):** 38.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 12 | 66.7% | 3 | 9 | 6 | 2.15% | 38.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 12 | 66.7% | 3 | 9 | 6 | 2.15% | 38.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 12 | 66.7% | 3 | 9 | 6 | 2.15% | 38.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 963.00 | 1108.39 | 1108.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 957.10 | 1104.07 | 1106.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1023.20 | 1022.36 | 1049.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 1023.20 | 1022.36 | 1049.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1048.15 | 1023.02 | 1049.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1045.95 | 1023.02 | 1049.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1045.00 | 1023.24 | 1048.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1050.45 | 1023.24 | 1048.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1050.00 | 1023.50 | 1049.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1102.75 | 1023.50 | 1049.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1092.00 | 1024.19 | 1049.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1092.60 | 1024.19 | 1049.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1073.60 | 1058.72 | 1062.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 1080.60 | 1058.72 | 1062.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1114.20 | 1059.38 | 1062.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1114.20 | 1059.38 | 1062.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1102.00 | 1059.80 | 1063.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1097.75 | 1059.80 | 1063.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 1086.45 | 1061.00 | 1063.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 13:15:00 | 1042.86 | 1060.89 | 1063.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1068.30 | 1060.96 | 1063.57 | SL hit (close>ema200) qty=0.50 sl=1060.96 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 1087.30 | 1055.97 | 1055.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 1092.50 | 1060.73 | 1058.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1063.90 | 1074.61 | 1067.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1063.90 | 1074.61 | 1067.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1063.90 | 1074.61 | 1067.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1063.90 | 1074.61 | 1067.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1060.40 | 1074.47 | 1067.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1060.40 | 1074.47 | 1067.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 978.10 | 1060.62 | 1060.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 974.00 | 1059.76 | 1060.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 911.50 | 897.87 | 945.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:45:00 | 911.30 | 897.87 | 945.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 924.15 | 898.36 | 945.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 907.85 | 899.78 | 944.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 862.46 | 898.59 | 942.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.30 | 894.28 | 935.45 | SL hit (close>ema200) qty=0.50 sl=894.28 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 925.00 | 891.32 | 891.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 10:15:00 | 934.85 | 892.46 | 891.88 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-30 09:15:00 | 1097.75 | 2025-09-30 13:15:00 | 1042.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1097.75 | 2025-09-30 14:15:00 | 1068.30 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2025-09-30 12:45:00 | 1086.45 | 2025-10-07 14:15:00 | 1032.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 12:45:00 | 1086.45 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2025-11-12 09:15:00 | 1071.00 | 2025-11-13 12:15:00 | 1087.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-11-13 10:15:00 | 1099.00 | 2025-11-13 12:15:00 | 1087.30 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2026-01-29 09:15:00 | 907.85 | 2026-01-29 14:15:00 | 862.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 907.85 | 2026-02-03 09:15:00 | 1001.30 | STOP_HIT | 0.50 | -10.29% |
| SELL | retest2 | 2026-02-11 14:00:00 | 913.05 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -7.80% |
| SELL | retest2 | 2026-02-12 14:30:00 | 902.75 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest2 | 2026-02-16 09:30:00 | 910.00 | 2026-03-02 09:15:00 | 864.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:45:00 | 919.50 | 2026-03-02 09:15:00 | 873.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 915.00 | 2026-03-02 09:15:00 | 869.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 09:30:00 | 910.00 | 2026-03-05 13:15:00 | 827.55 | TARGET_HIT | 0.50 | 9.06% |
| SELL | retest2 | 2026-02-23 13:45:00 | 919.50 | 2026-03-09 09:15:00 | 819.00 | TARGET_HIT | 0.50 | 10.93% |
| SELL | retest2 | 2026-02-25 15:15:00 | 915.00 | 2026-03-09 09:15:00 | 823.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-17 15:15:00 | 922.00 | 2026-04-29 14:15:00 | 925.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-20 09:30:00 | 922.80 | 2026-04-29 14:15:00 | 925.00 | STOP_HIT | 1.00 | -0.24% |
