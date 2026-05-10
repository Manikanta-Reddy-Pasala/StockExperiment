# Yes Bank Ltd. (YESBANK)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 22.90
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
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -1.70% / -1.73%
- **Sum % (uncompounded):** -15.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.70% | -15.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.70% | -15.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.70% | -15.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 20.45 | 17.95 | 17.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 21.78 | 18.70 | 18.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 20.44 | 20.51 | 19.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 20.44 | 20.51 | 19.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 20.08 | 20.43 | 19.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 20.20 | 20.25 | 19.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 20.14 | 20.24 | 19.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 20.25 | 20.24 | 19.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 20.16 | 20.26 | 19.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 19.96 | 20.22 | 19.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 19.99 | 20.22 | 19.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 19.85 | 20.19 | 19.91 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 19.98 | 20.13 | 19.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 19.97 | 20.13 | 19.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 19.89 | 20.14 | 19.95 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 19.89 | 20.14 | 19.95 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 19.97 | 20.12 | 19.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 19.70 | 20.12 | 19.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 19.70 | 20.12 | 19.95 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 19.97 | 20.11 | 19.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 19.38 | 20.06 | 19.93 | SL hit (close<static) qty=1.00 sl=19.43 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-01 14:15:00 | 18.61 | 19.82 | 19.82 | min_gap filter: gap=0.007% < 0.030% |
| TREND_RESET | 2025-08-01 14:15:00 | 18.61 | 19.82 | 19.82 | EMA inversion without crossover edge (EMA200=19.82 EMA400=19.82) — end cycle |
| CROSSOVER_SKIP | 2025-09-09 12:15:00 | 20.39 | 19.58 | 19.58 | min_gap filter: gap=0.006% < 0.030% |
| CROSSOVER_SKIP | 2025-12-30 14:15:00 | 21.39 | 21.98 | 21.98 | min_gap filter: gap=0.003% < 0.030% |
| CROSSOVER_SKIP | 2026-01-05 15:15:00 | 22.79 | 21.98 | 21.98 | min_gap filter: gap=0.004% < 0.030% |
| CROSSOVER_SKIP | 2026-01-29 10:15:00 | 21.16 | 22.09 | 22.09 | min_gap filter: gap=0.008% < 0.030% |
| CROSSOVER_SKIP | 2026-05-08 14:15:00 | 22.93 | 20.10 | 20.10 | min_gap filter: gap=0.027% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-25 12:45:00 | 20.20 | 2025-07-10 10:15:00 | 19.85 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-26 12:00:00 | 20.14 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-06-27 09:15:00 | 20.25 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-07-04 09:45:00 | 20.16 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-07-08 12:45:00 | 19.99 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-14 12:45:00 | 19.98 | 2025-07-23 09:15:00 | 19.89 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-14 15:00:00 | 19.97 | 2025-07-23 09:15:00 | 19.89 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-24 10:30:00 | 19.97 | 2025-07-24 11:15:00 | 19.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-24 14:45:00 | 19.97 | 2025-07-28 12:15:00 | 19.38 | STOP_HIT | 1.00 | -2.95% |
