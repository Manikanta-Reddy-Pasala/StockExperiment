# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 19
- **Target hits / Stop hits / Partials:** 7 / 23 / 8
- **Avg / median % per leg:** 1.16% / 1.19%
- **Sum % (uncompounded):** 44.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 34 | 15 | 44.1% | 3 | 23 | 8 | 0.12% | 4.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 15 | 44.1% | 3 | 23 | 8 | 0.12% | 4.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 19 | 50.0% | 7 | 23 | 8 | 1.16% | 44.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 987.50 | 1116.41 | 1116.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 975.60 | 1111.38 | 1114.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1023.20 | 1022.37 | 1051.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 1023.20 | 1022.37 | 1051.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1045.00 | 1023.25 | 1050.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1050.45 | 1023.25 | 1050.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1092.00 | 1024.20 | 1051.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1092.60 | 1024.20 | 1051.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1073.20 | 1024.69 | 1051.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1066.80 | 1058.43 | 1064.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 1072.55 | 1058.57 | 1064.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 1071.75 | 1058.72 | 1064.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 1114.20 | 1059.39 | 1064.28 | SL hit (close>static) qty=1.00 sl=1092.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 1080.30 | 1057.11 | 1057.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 1092.50 | 1060.73 | 1058.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1063.90 | 1074.62 | 1067.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1060.40 | 1074.47 | 1067.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1060.40 | 1074.47 | 1067.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 984.60 | 1061.45 | 1061.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 978.10 | 1060.62 | 1061.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 911.50 | 897.87 | 945.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:45:00 | 911.30 | 897.87 | 945.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 924.15 | 898.36 | 945.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 907.85 | 899.78 | 944.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 862.46 | 898.59 | 942.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.30 | 894.28 | 935.52 | SL hit (close>ema200) qty=0.50 sl=894.28 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 925.00 | 891.32 | 891.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 10:15:00 | 934.85 | 892.46 | 891.89 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 11:30:00 | 1086.20 | 2025-07-04 13:15:00 | 1192.84 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1091.00 | 2025-07-04 15:15:00 | 1194.82 | TARGET_HIT | 1.00 | 9.52% |
| BUY | retest2 | 2025-07-02 13:15:00 | 1084.40 | 2025-07-08 09:15:00 | 1200.10 | TARGET_HIT | 1.00 | 10.67% |
| BUY | retest2 | 2025-07-02 14:45:00 | 1108.00 | 2025-07-08 09:15:00 | 1218.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1066.80 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-09-26 10:00:00 | 1072.55 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-09-26 11:15:00 | 1071.75 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1071.50 | 2025-10-03 14:15:00 | 1072.40 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1054.40 | 2025-10-08 09:15:00 | 1017.92 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1056.80 | 2025-10-09 10:15:00 | 1007.09 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1060.10 | 2025-10-09 12:15:00 | 1003.96 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-10-07 10:00:00 | 1056.40 | 2025-10-09 12:15:00 | 1003.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1054.40 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1056.80 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1060.10 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-10-07 10:00:00 | 1056.40 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2025-10-17 14:15:00 | 1033.90 | 2025-10-23 09:15:00 | 1075.00 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-10-20 09:15:00 | 1028.80 | 2025-10-23 09:15:00 | 1075.00 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-10-24 13:45:00 | 1036.20 | 2025-10-27 14:15:00 | 1062.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-10-27 09:30:00 | 1035.00 | 2025-10-28 09:15:00 | 1066.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-27 12:15:00 | 1046.60 | 2025-10-28 09:15:00 | 1066.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1040.50 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-10-29 11:15:00 | 1045.80 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-29 12:00:00 | 1045.00 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1041.00 | 2025-11-11 14:15:00 | 1079.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-11-11 09:45:00 | 1025.60 | 2025-11-11 14:15:00 | 1079.00 | STOP_HIT | 1.00 | -5.21% |
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
