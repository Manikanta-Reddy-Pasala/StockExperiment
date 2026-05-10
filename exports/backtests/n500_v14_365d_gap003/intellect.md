# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 808.00
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
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 2
- **Avg / median % per leg:** 0.42% / -0.31%
- **Sum % (uncompounded):** 3.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 0 | 6 | 2 | 0.42% | 3.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 3 | 37.5% | 0 | 6 | 2 | 0.42% | 3.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 3 | 37.5% | 0 | 6 | 2 | 0.42% | 3.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 933.50 | 1055.30 | 1055.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 926.90 | 1049.63 | 1052.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 993.40 | 980.32 | 1006.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 993.40 | 980.32 | 1006.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1007.35 | 981.01 | 1006.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1007.35 | 981.01 | 1006.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1010.25 | 981.30 | 1006.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1013.05 | 981.30 | 1006.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1017.80 | 981.67 | 1006.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1017.80 | 981.67 | 1006.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1018.00 | 982.03 | 1006.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1011.35 | 982.03 | 1006.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:30:00 | 1010.20 | 982.58 | 1006.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1008.40 | 983.93 | 1006.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1006.68 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1006.68 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1006.68 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 1013.00 | 986.70 | 1007.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1020.85 | 987.04 | 1007.25 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 1009.10 | 998.43 | 1007.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 1009.10 | 998.43 | 1007.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1008.90 | 998.53 | 1007.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 993.95 | 998.86 | 1007.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 12:15:00 | 944.25 | 996.71 | 1005.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 997.05 | 995.53 | 1005.15 | SL hit (close>ema200) qty=0.50 sl=995.53 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:00:00 | 1000.15 | 995.54 | 1005.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:15:00 | 950.14 | 991.44 | 1002.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 992.55 | 986.01 | 998.20 | SL hit (close>ema200) qty=0.50 sl=986.01 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-03 09:15:00 | 1206.40 | 1007.66 | 1007.62 | min_gap filter: gap=0.004% < 0.030% |
| TREND_RESET | 2025-11-03 09:15:00 | 1206.40 | 1007.66 | 1007.62 | EMA inversion without crossover edge (EMA200=1007.66 EMA400=1007.62) — end cycle |
| CROSSOVER_SKIP | 2025-12-26 09:15:00 | 1007.50 | 1050.16 | 1050.29 | min_gap filter: gap=0.014% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-11 09:15:00 | 1011.35 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-11 10:30:00 | 1010.20 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-09-12 09:15:00 | 1008.40 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-15 09:15:00 | 1013.00 | 2025-09-15 09:15:00 | 1020.85 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-13 09:15:00 | 993.95 | 2025-10-14 12:15:00 | 944.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 993.95 | 2025-10-15 09:15:00 | 997.05 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2025-10-15 12:00:00 | 1000.15 | 2025-10-20 09:15:00 | 950.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 12:00:00 | 1000.15 | 2025-10-24 13:15:00 | 992.55 | STOP_HIT | 0.50 | 0.76% |
