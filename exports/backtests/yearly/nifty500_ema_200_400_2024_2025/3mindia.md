# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 32070.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 47 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 43
- **Target hits / Stop hits / Partials:** 3 / 48 / 7
- **Avg / median % per leg:** -0.63% / -1.28%
- **Sum % (uncompounded):** -36.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 9 | 20.9% | 0 | 39 | 4 | -0.69% | -29.8% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| BUY @ 3rd Alert (retest2) | 35 | 1 | 2.9% | 0 | 35 | 0 | -1.59% | -55.7% |
| SELL (all) | 15 | 6 | 40.0% | 3 | 9 | 3 | -0.45% | -6.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 3 | 9 | 3 | -0.45% | -6.7% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| retest2 (combined) | 50 | 7 | 14.0% | 3 | 44 | 3 | -1.25% | -62.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 11:15:00 | 33481.40 | 30593.21 | 30584.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 14:15:00 | 33916.95 | 30679.87 | 30628.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 10:15:00 | 37413.25 | 37447.72 | 35512.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 11:00:00 | 37413.25 | 37447.72 | 35512.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 35999.85 | 37943.72 | 36806.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:00:00 | 35999.85 | 37943.72 | 36806.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 36149.85 | 37925.87 | 36802.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 36521.70 | 37925.87 | 36802.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 35776.85 | 37853.91 | 36788.75 | SL hit (close<static) qty=1.00 sl=35851.05 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 34931.25 | 36208.51 | 36213.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 34456.95 | 35636.57 | 35856.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 11:15:00 | 35107.70 | 35092.72 | 35495.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:00:00 | 35107.70 | 35092.72 | 35495.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 34700.00 | 34288.59 | 34855.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:30:00 | 33680.00 | 34859.97 | 35033.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 11:00:00 | 33706.05 | 34848.49 | 35026.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:00:00 | 33636.45 | 34811.95 | 35005.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 31996.00 | 34635.11 | 34906.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 32020.75 | 34635.11 | 34906.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 31954.63 | 34635.11 | 34906.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 09:15:00 | 30312.00 | 31648.49 | 32613.42 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 29850.00 | 28750.17 | 28747.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 30000.00 | 28762.61 | 28753.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 28750.00 | 29289.15 | 29069.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 28750.00 | 29289.15 | 29069.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 28750.00 | 29289.15 | 29069.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 29200.00 | 29270.49 | 29065.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 12:30:00 | 29375.00 | 29277.68 | 29074.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 29625.00 | 29275.08 | 29076.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 29280.00 | 29288.96 | 29090.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 29130.00 | 29293.18 | 29097.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 29130.00 | 29293.18 | 29097.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 29295.00 | 29293.19 | 29098.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 29380.00 | 29292.96 | 29099.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 29355.00 | 29291.78 | 29102.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 28850.00 | 29581.39 | 29319.53 | SL hit (close<static) qty=1.00 sl=29090.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 28365.00 | 29244.88 | 29245.09 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 29670.00 | 29171.44 | 29170.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 29970.00 | 29184.59 | 29177.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 30215.00 | 30310.71 | 29893.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:00:00 | 30215.00 | 30310.71 | 29893.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 30405.00 | 30578.96 | 30217.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 30330.00 | 30578.96 | 30217.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 30380.00 | 30689.49 | 30344.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 30890.00 | 30666.54 | 30345.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 30585.00 | 30739.53 | 30445.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 30560.00 | 30736.12 | 30446.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 30640.00 | 30725.95 | 30448.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 30500.00 | 30722.85 | 30449.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 30500.00 | 30722.85 | 30449.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 30405.00 | 30717.20 | 30449.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 30405.00 | 30717.20 | 30449.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 30360.00 | 30713.65 | 30449.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 30385.00 | 30713.65 | 30449.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 30195.00 | 30708.49 | 30447.97 | SL hit (close<static) qty=1.00 sl=30260.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 28975.00 | 30258.44 | 30263.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 28805.00 | 30231.58 | 30249.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 29645.00 | 29638.07 | 29872.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:45:00 | 29670.00 | 29638.07 | 29872.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 29735.00 | 29640.28 | 29871.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 29700.00 | 29640.28 | 29871.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 29800.00 | 29643.08 | 29867.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 29860.00 | 29643.08 | 29867.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 29715.00 | 29645.74 | 29867.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 29885.00 | 29645.74 | 29867.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 29645.00 | 29645.41 | 29859.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 29520.00 | 29645.41 | 29859.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 30055.00 | 29650.57 | 29854.51 | SL hit (close>static) qty=1.00 sl=29900.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 36045.00 | 30033.74 | 30009.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 36375.00 | 30209.28 | 30098.05 | Break + close above crossover candle high |
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

### Cycle 8 — SELL (started 2026-03-17 13:15:00)

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
| SELL | retest2 | 2024-05-21 09:15:00 | 30791.00 | 2024-05-28 14:15:00 | 34200.05 | STOP_HIT | 1.00 | -11.07% |
| SELL | retest2 | 2024-05-21 09:45:00 | 31025.05 | 2024-05-28 14:15:00 | 34200.05 | STOP_HIT | 1.00 | -10.23% |
| SELL | retest2 | 2024-05-23 12:30:00 | 31001.25 | 2024-05-28 14:15:00 | 34200.05 | STOP_HIT | 1.00 | -10.32% |
| BUY | retest2 | 2024-08-16 09:15:00 | 36521.70 | 2024-08-16 12:15:00 | 35776.85 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-08-19 09:15:00 | 36346.90 | 2024-08-19 12:15:00 | 35850.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-08-19 10:15:00 | 36161.00 | 2024-08-19 12:15:00 | 35850.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-19 11:00:00 | 36196.00 | 2024-08-19 12:15:00 | 35850.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-11-13 09:30:00 | 33680.00 | 2024-11-18 09:15:00 | 31996.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 11:00:00 | 33706.05 | 2024-11-18 09:15:00 | 32020.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 14:00:00 | 33636.45 | 2024-11-18 09:15:00 | 31954.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 09:30:00 | 33680.00 | 2024-12-30 09:15:00 | 30312.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-13 11:00:00 | 33706.05 | 2024-12-30 09:15:00 | 30335.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-13 14:00:00 | 33636.45 | 2024-12-30 09:15:00 | 30272.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-09 15:15:00 | 29200.00 | 2025-05-28 14:15:00 | 28850.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-12 12:30:00 | 29375.00 | 2025-05-28 14:15:00 | 28850.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-05-13 09:15:00 | 29625.00 | 2025-06-02 09:15:00 | 29010.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-05-14 09:15:00 | 29280.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-05-15 09:15:00 | 29380.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-05-15 13:00:00 | 29355.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-05-30 12:30:00 | 29350.00 | 2025-06-11 13:15:00 | 29300.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-06-02 10:45:00 | 29380.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-06-09 09:15:00 | 29455.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-09 13:00:00 | 29400.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-09 15:15:00 | 29400.00 | 2025-06-11 15:15:00 | 29265.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-10 12:45:00 | 29400.00 | 2025-06-13 12:15:00 | 29025.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-11 09:15:00 | 29515.00 | 2025-06-23 14:15:00 | 28365.00 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-06-11 09:45:00 | 29510.00 | 2025-06-23 14:15:00 | 28365.00 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2025-06-11 10:15:00 | 29505.00 | 2025-06-23 14:15:00 | 28365.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2025-06-11 11:15:00 | 29550.00 | 2025-06-23 14:15:00 | 28365.00 | STOP_HIT | 1.00 | -4.01% |
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
