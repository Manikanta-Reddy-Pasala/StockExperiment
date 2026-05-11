# The Ramco Cements Ltd. (RAMCOCEM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3162 bars)
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
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 16 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 1 / 12 / 5
- **Avg / median % per leg:** 0.53% / -1.98%
- **Sum % (uncompounded):** 9.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 1 | 12 | 1 | -0.75% | -10.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.09% | 6.2% |
| BUY @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 1 | 11 | 0 | -1.39% | -16.7% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 0 | 4 | 5.00% | 20.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 0 | 4 | 5.00% | 20.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.09% | 6.2% |
| retest2 (combined) | 16 | 5 | 31.2% | 1 | 11 | 4 | 0.20% | 3.3% |

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

### Cycle 2 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 1009.90 | 1037.83 | 1037.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1008.50 | 1037.53 | 1037.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 1031.30 | 1031.15 | 1034.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 1031.30 | 1031.15 | 1034.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1037.50 | 1031.22 | 1034.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 1037.50 | 1031.22 | 1034.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1037.20 | 1031.28 | 1034.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1038.80 | 1031.28 | 1034.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1023.70 | 1023.58 | 1029.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 1026.50 | 1023.58 | 1029.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1027.00 | 1023.48 | 1029.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 1032.80 | 1023.48 | 1029.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1026.80 | 1023.51 | 1029.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 1028.60 | 1023.51 | 1029.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1028.30 | 1023.56 | 1029.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 1029.30 | 1023.56 | 1029.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1028.90 | 1023.61 | 1029.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 1029.10 | 1023.61 | 1029.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1052.50 | 1023.92 | 1029.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 1052.50 | 1023.92 | 1029.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1050.50 | 1033.66 | 1033.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1056.50 | 1033.89 | 1033.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1057.50 | 1060.55 | 1050.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:45:00 | 1055.70 | 1060.55 | 1050.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1068.00 | 1061.18 | 1051.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 10:00:00 | 1074.60 | 1061.31 | 1051.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1071.00 | 1061.41 | 1051.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 12:00:00 | 1077.20 | 1061.56 | 1051.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 1046.00 | 1063.61 | 1053.79 | SL hit (close<static) qty=1.00 sl=1046.50 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 970.30 | 1085.34 | 1085.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 946.70 | 1078.86 | 1082.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 987.00 | 985.61 | 1023.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:30:00 | 985.60 | 985.61 | 1023.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1003.20 | 986.90 | 1018.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 996.30 | 987.20 | 1017.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 998.00 | 987.58 | 1017.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 998.00 | 991.97 | 1016.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:45:00 | 995.70 | 992.05 | 1016.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 946.48 | 984.93 | 1009.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 948.10 | 984.93 | 1009.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 948.10 | 984.93 | 1009.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 945.91 | 984.93 | 1009.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 09:15:00 | 950.00 | 2025-05-14 13:15:00 | 997.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-12 09:15:00 | 950.00 | 2025-05-28 09:15:00 | 961.25 | STOP_HIT | 0.50 | 1.18% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1062.00 | 2025-10-30 09:15:00 | 1037.85 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-10-29 10:00:00 | 1058.75 | 2025-10-30 09:15:00 | 1037.85 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-10-29 13:45:00 | 1060.35 | 2025-10-30 09:15:00 | 1037.85 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-10-29 14:30:00 | 1058.80 | 2025-10-30 09:15:00 | 1037.85 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1060.45 | 2025-10-31 15:15:00 | 1035.20 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-10-30 15:00:00 | 1058.75 | 2025-10-31 15:15:00 | 1035.20 | STOP_HIT | 1.00 | -2.22% |
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
