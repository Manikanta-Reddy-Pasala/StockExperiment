# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 32070.00
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
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 11 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 0 / 15 / 4
- **Avg / median % per leg:** 0.32% / -0.79%
- **Sum % (uncompounded):** 6.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 8 | 42.1% | 0 | 15 | 4 | 0.32% | 6.1% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.80% | -19.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.80% | -19.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 36045.00 | 30033.74 | 30009.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 36375.00 | 30209.28 | 30098.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 34040.00 | 34147.01 | 32904.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 34500.00 | 34150.52 | 32912.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:00:00 | 34265.00 | 34188.47 | 32980.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:45:00 | 34320.00 | 34190.08 | 32987.12 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:45:00 | 34320.00 | 34218.04 | 33083.01 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 35978.25 | 34729.57 | 33855.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 36036.00 | 34729.57 | 33855.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 36036.00 | 34729.57 | 33855.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:15:00 | 36225.00 | 34780.55 | 33898.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 34860.00 | 34993.10 | 34095.50 | SL hit (close<ema200) qty=0.50 sl=34993.10 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 34860.00 | 34993.10 | 34095.50 | SL hit (close<ema200) qty=0.50 sl=34993.10 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 34860.00 | 34993.10 | 34095.50 | SL hit (close<ema200) qty=0.50 sl=34993.10 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 34860.00 | 34993.10 | 34095.50 | SL hit (close<ema200) qty=0.50 sl=34993.10 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 34435.00 | 34968.16 | 34143.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 34275.00 | 34968.16 | 34143.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 34045.00 | 34944.00 | 34151.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 34045.00 | 34944.00 | 34151.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 34090.00 | 34935.50 | 34151.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 33830.00 | 34935.50 | 34151.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 34120.00 | 34927.38 | 34150.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 33885.00 | 34927.38 | 34150.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 33835.00 | 34916.51 | 34149.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:00:00 | 33835.00 | 34916.51 | 34149.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 34230.00 | 34891.58 | 34148.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:30:00 | 34300.00 | 34884.25 | 34148.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 34815.00 | 34884.25 | 34148.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 14:30:00 | 34510.00 | 34860.72 | 34161.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:15:00 | 34330.00 | 34874.74 | 34248.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 34160.00 | 34867.63 | 34248.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 34160.00 | 34867.63 | 34248.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 34060.00 | 34859.59 | 34247.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 34060.00 | 34859.59 | 34247.48 | SL hit (close<static) qty=1.00 sl=34070.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 34060.00 | 34859.59 | 34247.48 | SL hit (close<static) qty=1.00 sl=34070.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 34060.00 | 34859.59 | 34247.48 | SL hit (close<static) qty=1.00 sl=34070.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 34060.00 | 34859.59 | 34247.48 | SL hit (close<static) qty=1.00 sl=34070.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 34060.00 | 34859.59 | 34247.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 33865.00 | 34752.11 | 34227.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 33865.00 | 34752.11 | 34227.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 34440.00 | 34397.74 | 34117.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:15:00 | 34690.00 | 34396.98 | 34120.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:00:00 | 34715.00 | 34401.95 | 34126.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 34095.00 | 34397.97 | 34129.62 | SL hit (close<static) qty=1.00 sl=34100.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 34095.00 | 34397.97 | 34129.62 | SL hit (close<static) qty=1.00 sl=34100.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 34740.00 | 34401.73 | 34135.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 34035.00 | 34400.03 | 34137.31 | SL hit (close<static) qty=1.00 sl=34100.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:15:00 | 34850.00 | 34391.19 | 34139.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 34800.00 | 35241.21 | 34687.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:45:00 | 34150.00 | 35241.21 | 34687.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 34770.00 | 35236.52 | 34688.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 35320.00 | 35231.83 | 34688.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 34820.00 | 35802.42 | 35211.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 14:15:00 | 34160.00 | 35786.08 | 35205.94 | SL hit (close<static) qty=1.00 sl=34550.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 14:15:00 | 34160.00 | 35786.08 | 35205.94 | SL hit (close<static) qty=1.00 sl=34550.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:30:00 | 34885.00 | 35786.08 | 35205.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 34300.00 | 35771.29 | 35201.43 | SL hit (close<static) qty=1.00 sl=34550.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 33990.00 | 35723.20 | 35185.69 | SL hit (close<static) qty=1.00 sl=34100.00 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-17 13:15:00 | 33260.00 | 34784.25 | 34791.79 | min_gap filter: gap=0.023% < 0.030% |
| TREND_RESET | 2026-03-17 13:15:00 | 33260.00 | 34784.25 | 34791.79 | EMA inversion without crossover edge (EMA200=34784.25 EMA400=34791.79) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-08 10:00:00 | 34500.00 | 2026-01-01 13:15:00 | 35978.25 | PARTIAL | 0.50 | 4.28% |
| BUY | retest1 | 2025-12-09 11:00:00 | 34265.00 | 2026-01-01 13:15:00 | 36036.00 | PARTIAL | 0.50 | 5.17% |
| BUY | retest1 | 2025-12-09 11:45:00 | 34320.00 | 2026-01-01 13:15:00 | 36036.00 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-11 11:45:00 | 34320.00 | 2026-01-02 10:15:00 | 36225.00 | PARTIAL | 0.50 | 5.55% |
| BUY | retest1 | 2025-12-08 10:00:00 | 34500.00 | 2026-01-07 09:15:00 | 34860.00 | STOP_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2025-12-09 11:00:00 | 34265.00 | 2026-01-07 09:15:00 | 34860.00 | STOP_HIT | 0.50 | 1.74% |
| BUY | retest1 | 2025-12-09 11:45:00 | 34320.00 | 2026-01-07 09:15:00 | 34860.00 | STOP_HIT | 0.50 | 1.57% |
| BUY | retest1 | 2025-12-11 11:45:00 | 34320.00 | 2026-01-07 09:15:00 | 34860.00 | STOP_HIT | 0.50 | 1.57% |
| BUY | retest2 | 2026-01-12 14:30:00 | 34300.00 | 2026-01-20 12:15:00 | 34060.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-01-12 15:15:00 | 34815.00 | 2026-01-20 12:15:00 | 34060.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-01-13 14:30:00 | 34510.00 | 2026-01-20 12:15:00 | 34060.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-01-20 11:15:00 | 34330.00 | 2026-01-20 12:15:00 | 34060.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-01-30 13:15:00 | 34690.00 | 2026-02-01 11:15:00 | 34095.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-30 15:00:00 | 34715.00 | 2026-02-01 11:15:00 | 34095.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-02-01 14:30:00 | 34740.00 | 2026-02-02 09:15:00 | 34035.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-02-02 15:15:00 | 34850.00 | 2026-03-04 14:15:00 | 34160.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-16 12:15:00 | 35320.00 | 2026-03-04 14:15:00 | 34160.00 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2026-03-04 14:00:00 | 34820.00 | 2026-03-04 15:15:00 | 34300.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-03-04 14:30:00 | 34885.00 | 2026-03-05 11:15:00 | 33990.00 | STOP_HIT | 1.00 | -2.57% |
