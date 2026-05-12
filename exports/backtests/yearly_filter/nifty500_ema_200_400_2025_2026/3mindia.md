# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 32070.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 36 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 36
- **Target hits / Stop hits / Partials:** 0 / 40 / 4
- **Avg / median % per leg:** -0.94% / -1.30%
- **Sum % (uncompounded):** -41.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 8 | 21.1% | 0 | 34 | 4 | -0.56% | -21.5% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| BUY @ 3rd Alert (retest2) | 30 | 0 | 0.0% | 0 | 30 | 0 | -1.58% | -47.4% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.35% | -20.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.35% | -20.1% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| retest2 (combined) | 36 | 0 | 0.0% | 0 | 36 | 0 | -1.87% | -67.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 28390.00 | 29236.38 | 29240.55 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 29670.00 | 29171.44 | 29170.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 29970.00 | 29184.59 | 29177.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 30215.00 | 30310.71 | 29893.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:00:00 | 30215.00 | 30310.71 | 29893.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 30405.00 | 30578.96 | 30217.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 30330.00 | 30578.96 | 30217.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 30380.00 | 30689.49 | 30344.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 30890.00 | 30666.54 | 30345.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 30585.00 | 30739.53 | 30445.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 30560.00 | 30736.12 | 30446.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 30640.00 | 30725.95 | 30448.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 30500.00 | 30722.85 | 30449.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 30500.00 | 30722.85 | 30449.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 30405.00 | 30717.20 | 30449.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 30405.00 | 30717.20 | 30449.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 30360.00 | 30713.65 | 30449.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 30385.00 | 30713.65 | 30449.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 30195.00 | 30708.49 | 30447.93 | SL hit (close<static) qty=1.00 sl=30260.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 28975.00 | 30258.44 | 30263.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 28805.00 | 30231.58 | 30249.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 29645.00 | 29638.07 | 29872.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:45:00 | 29670.00 | 29638.07 | 29872.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 29735.00 | 29640.28 | 29871.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 29700.00 | 29640.28 | 29871.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 29800.00 | 29643.08 | 29867.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 29860.00 | 29643.08 | 29867.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 29715.00 | 29645.74 | 29867.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 29885.00 | 29645.74 | 29867.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 29645.00 | 29645.41 | 29859.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 29520.00 | 29645.41 | 29859.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 30055.00 | 29650.57 | 29854.49 | SL hit (close>static) qty=1.00 sl=29900.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 36045.00 | 30033.74 | 30009.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 36375.00 | 30209.28 | 30098.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 34040.00 | 34147.01 | 32904.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 34500.00 | 34150.52 | 32912.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:00:00 | 34265.00 | 34188.47 | 32980.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:45:00 | 34320.00 | 34190.08 | 32987.12 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:45:00 | 34320.00 | 34218.04 | 33083.00 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 35978.25 | 34729.57 | 33855.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 36036.00 | 34729.57 | 33855.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 36036.00 | 34729.57 | 33855.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:15:00 | 36225.00 | 34780.55 | 33898.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 34860.00 | 34993.10 | 34095.50 | SL hit (close<ema200) qty=0.50 sl=34993.10 alert=retest1 |

### Cycle 5 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 33260.00 | 34784.25 | 34791.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 33080.00 | 34767.29 | 34783.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.38 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 12:30:00 | 29375.00 | 2025-05-28 14:15:00 | 28850.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-05-13 09:15:00 | 29625.00 | 2025-05-28 14:15:00 | 28850.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-05-14 09:15:00 | 29280.00 | 2025-06-02 09:15:00 | 29010.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-05-15 09:15:00 | 29380.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-05-15 13:00:00 | 29355.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-05-30 12:30:00 | 29350.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-06-02 10:45:00 | 29380.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-09 09:15:00 | 29455.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-09 13:00:00 | 29400.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-09 15:15:00 | 29400.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-10 12:45:00 | 29400.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-11 09:15:00 | 29515.00 | 2025-06-13 12:15:00 | 29025.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-11 09:45:00 | 29510.00 | 2025-06-23 15:15:00 | 28390.00 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2025-06-11 10:15:00 | 29505.00 | 2025-06-23 15:15:00 | 28390.00 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-06-11 11:15:00 | 29550.00 | 2025-06-23 15:15:00 | 28390.00 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2025-09-08 09:15:00 | 30890.00 | 2025-09-17 13:15:00 | 30195.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-09-15 15:15:00 | 30585.00 | 2025-09-17 13:15:00 | 30195.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-16 10:15:00 | 30560.00 | 2025-09-17 13:15:00 | 30195.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-09-16 15:15:00 | 30640.00 | 2025-09-17 13:15:00 | 30195.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-24 10:15:00 | 29520.00 | 2025-10-27 09:15:00 | 30055.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-28 14:45:00 | 29510.00 | 2025-11-03 09:15:00 | 30550.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-10-29 11:00:00 | 29440.00 | 2025-11-03 09:15:00 | 30550.00 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-10-30 14:00:00 | 29465.00 | 2025-11-03 09:15:00 | 30550.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-10-31 10:15:00 | 29485.00 | 2025-11-03 09:15:00 | 30550.00 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-10-31 13:00:00 | 29465.00 | 2025-11-03 09:15:00 | 30550.00 | STOP_HIT | 1.00 | -3.68% |
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
