# The Ramco Cements Ltd. (RAMCOCEM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 953.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 36 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 15 / 29
- **Target hits / Stop hits / Partials:** 1 / 34 / 9
- **Avg / median % per leg:** -0.30% / -1.85%
- **Sum % (uncompounded):** -13.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 3 | 16.7% | 1 | 16 | 1 | -0.82% | -14.7% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.09% | 6.2% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -1.30% | -20.9% |
| SELL (all) | 26 | 12 | 46.2% | 0 | 18 | 8 | 0.05% | 1.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.88% | -5.8% |
| SELL @ 3rd Alert (retest2) | 24 | 12 | 50.0% | 0 | 16 | 8 | 0.29% | 7.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.11% | 0.4% |
| retest2 (combined) | 40 | 13 | 32.5% | 1 | 31 | 8 | -0.35% | -13.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 09:15:00 | 950.00 | 937.23 | 920.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 13:15:00 | 997.50 | 943.26 | 924.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 961.25 | 965.64 | 943.35 | SL hit (close<ema200) qty=0.50 sl=965.64 alert=retest1 |

### Cycle 2 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 1045.00 | 1075.08 | 1075.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1024.10 | 1063.97 | 1068.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1030.40 | 1027.25 | 1044.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:15:00 | 1014.25 | 1027.45 | 1043.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1013.85 | 1027.07 | 1043.06 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1043.25 | 1026.88 | 1042.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1043.25 | 1026.88 | 1042.33 | SL hit (close>ema400) qty=1.00 sl=1042.33 alert=retest1 |

### Cycle 3 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 1061.40 | 1030.42 | 1030.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1074.70 | 1033.96 | 1032.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1057.50 | 1059.04 | 1048.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:45:00 | 1055.70 | 1059.04 | 1048.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1068.00 | 1059.86 | 1049.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 10:00:00 | 1074.60 | 1060.00 | 1049.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1071.00 | 1060.11 | 1049.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 12:00:00 | 1077.20 | 1060.28 | 1049.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 1046.00 | 1062.54 | 1051.87 | SL hit (close<static) qty=1.00 sl=1046.50 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 978.80 | 1084.18 | 1084.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 946.70 | 1078.76 | 1081.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 987.00 | 985.57 | 1022.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:30:00 | 985.60 | 985.57 | 1022.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1003.20 | 986.87 | 1017.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 996.30 | 987.18 | 1017.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 998.00 | 987.55 | 1017.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 998.00 | 991.95 | 1016.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:45:00 | 995.70 | 992.03 | 1016.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 946.48 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 948.10 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 948.10 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 945.91 | 984.91 | 1008.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 09:15:00 | 950.00 | 2025-05-14 13:15:00 | 997.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-12 09:15:00 | 950.00 | 2025-05-28 09:15:00 | 961.25 | STOP_HIT | 0.50 | 1.18% |
| BUY | retest2 | 2025-08-18 09:30:00 | 1085.50 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-08-18 10:00:00 | 1089.90 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-08-18 15:15:00 | 1082.80 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-08-19 13:15:00 | 1084.00 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-08-22 13:30:00 | 1077.90 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-08-22 15:15:00 | 1079.90 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-25 09:30:00 | 1080.00 | 2025-08-25 12:15:00 | 1063.30 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-09-02 11:00:00 | 1077.60 | 2025-09-02 13:15:00 | 1069.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1086.00 | 2025-09-05 15:15:00 | 1066.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-04 13:30:00 | 1086.10 | 2025-09-05 15:15:00 | 1066.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest1 | 2025-10-17 09:15:00 | 1014.25 | 2025-10-21 13:15:00 | 1043.25 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest1 | 2025-10-20 09:15:00 | 1013.85 | 2025-10-21 13:15:00 | 1043.25 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-11-03 11:45:00 | 1029.90 | 2025-11-14 09:15:00 | 978.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1030.50 | 2025-11-14 09:15:00 | 978.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1027.00 | 2025-11-14 09:15:00 | 975.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:45:00 | 1024.00 | 2025-11-14 09:15:00 | 972.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 11:45:00 | 1029.90 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1030.50 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1027.00 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 0.99% |
| SELL | retest2 | 2025-11-06 09:45:00 | 1024.00 | 2025-11-21 10:15:00 | 1016.80 | STOP_HIT | 0.50 | 0.70% |
| SELL | retest2 | 2025-11-21 15:00:00 | 1009.90 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-11-24 09:30:00 | 1012.50 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-11-24 13:30:00 | 1014.00 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-11-26 13:15:00 | 1014.40 | 2025-11-28 14:15:00 | 1037.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1025.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-01 14:30:00 | 1025.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1019.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-12-03 14:45:00 | 1025.00 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-04 10:00:00 | 1011.80 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-12-04 11:45:00 | 1011.50 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-12-04 13:45:00 | 1012.20 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1011.30 | 2025-12-12 09:15:00 | 1052.50 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2026-01-21 10:00:00 | 1074.60 | 2026-01-23 15:15:00 | 1046.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-01-21 11:00:00 | 1071.00 | 2026-01-23 15:15:00 | 1046.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-21 12:00:00 | 1077.20 | 2026-01-23 15:15:00 | 1046.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-01-27 15:00:00 | 1073.20 | 2026-02-09 09:15:00 | 1180.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-02 13:45:00 | 1105.70 | 2026-03-04 09:15:00 | 1066.30 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2026-03-05 15:15:00 | 1106.90 | 2026-03-06 10:15:00 | 1081.20 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-04-16 11:45:00 | 996.30 | 2026-04-28 15:15:00 | 946.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 15:15:00 | 998.00 | 2026-04-28 15:15:00 | 948.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 15:15:00 | 998.00 | 2026-04-28 15:15:00 | 948.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:45:00 | 995.70 | 2026-04-28 15:15:00 | 945.91 | PARTIAL | 0.50 | 5.00% |
