# Divi's Laboratories Ltd. (DIVISLAB)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 6705.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.43% / 2.86%
- **Sum % (uncompounded):** 1.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.43% | 1.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.43% | 1.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.43% | 1.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 5980.00 | 6491.92 | 6494.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 5959.50 | 6462.34 | 6479.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 6260.00 | 6145.58 | 6248.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 6240.00 | 6146.52 | 6248.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 6249.50 | 6146.52 | 6248.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 6213.00 | 6148.08 | 6247.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 6203.50 | 6148.66 | 6247.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 5893.32 | 6121.62 | 6220.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 6026.00 | 5993.88 | 6128.03 | SL hit (close>ema200) qty=0.50 sl=5993.88 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 6200.00 | 6004.04 | 6128.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 6191.00 | 6013.08 | 6129.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6386.00 | 6025.97 | 6131.46 | SL hit (close>static) qty=1.00 sl=6254.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6386.00 | 6025.97 | 6131.46 | SL hit (close>static) qty=1.00 sl=6254.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | min_gap filter: gap=0.012% < 0.030% |
| TREND_RESET | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | EMA inversion without crossover edge (EMA200=6217.42 EMA400=6216.63) — end cycle |
| CROSSOVER_SKIP | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6412.45 | min_gap filter: gap=0.021% < 0.030% |
| CROSSOVER_SKIP | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | min_gap filter: gap=0.002% < 0.030% |
| CROSSOVER_SKIP | 2026-03-12 09:15:00 | 6233.00 | 6294.15 | 6294.33 | min_gap filter: gap=0.003% < 0.030% |
| CROSSOVER_SKIP | 2026-04-28 13:15:00 | 6444.50 | 6198.32 | 6198.26 | min_gap filter: gap=0.001% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-19 13:45:00 | 6203.50 | 2025-09-25 14:15:00 | 5893.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 13:45:00 | 6203.50 | 2025-10-07 11:15:00 | 6026.00 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-10-08 12:00:00 | 6200.00 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-09 10:30:00 | 6191.00 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -3.15% |
