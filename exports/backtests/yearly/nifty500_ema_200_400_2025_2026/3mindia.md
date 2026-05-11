# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 32070.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 22 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 22
- **Target hits / Stop hits / Partials:** 0 / 26 / 4
- **Avg / median % per leg:** -0.33% / -1.30%
- **Sum % (uncompounded):** -9.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 8 | 26.7% | 0 | 26 | 4 | -0.33% | -9.8% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| BUY @ 3rd Alert (retest2) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.62% | -35.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 25.9% |
| retest2 (combined) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.62% | -35.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 31740.00 | 29791.33 | 29782.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 11:15:00 | 31815.00 | 30196.26 | 30006.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 30215.00 | 30320.36 | 30096.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:00:00 | 30215.00 | 30320.36 | 30096.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 30595.00 | 30322.20 | 30100.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 30595.00 | 30322.20 | 30100.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 30195.00 | 30534.25 | 30250.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 30195.00 | 30534.25 | 30250.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 30370.00 | 30532.62 | 30250.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 30285.00 | 30532.62 | 30250.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 30475.00 | 30523.32 | 30254.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 30255.00 | 30523.32 | 30254.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 30360.00 | 30516.72 | 30260.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 30205.00 | 30516.72 | 30260.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 30505.00 | 30592.96 | 30335.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 30590.00 | 30592.96 | 30335.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 30555.00 | 30592.41 | 30337.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 30590.00 | 30592.41 | 30337.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 30630.00 | 30593.07 | 30341.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 30465.00 | 30592.16 | 30343.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 30320.00 | 30592.16 | 30343.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 30405.00 | 30582.89 | 30347.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 30330.00 | 30582.89 | 30347.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 30300.00 | 30579.19 | 30352.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 30285.00 | 30579.19 | 30352.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 30270.00 | 30576.12 | 30352.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 30560.00 | 30574.41 | 30352.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:00:00 | 30740.00 | 30576.06 | 30354.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 30535.00 | 30692.09 | 30450.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 30185.00 | 30681.76 | 30448.99 | SL hit (close<static) qty=1.00 sl=30215.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 29045.00 | 30381.38 | 30385.02 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 35820.00 | 30091.47 | 30064.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 36375.00 | 30209.44 | 30123.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 34040.00 | 34147.05 | 32916.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 34500.00 | 34150.56 | 32924.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:00:00 | 34265.00 | 34188.50 | 32991.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:45:00 | 34320.00 | 34190.11 | 32998.48 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:45:00 | 34320.00 | 34218.07 | 33093.60 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 35978.25 | 34729.58 | 33862.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 36036.00 | 34729.58 | 33862.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:15:00 | 36036.00 | 34729.58 | 33862.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:15:00 | 36225.00 | 34780.56 | 33905.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 34860.00 | 34993.10 | 34101.20 | SL hit (close<ema200) qty=0.50 sl=34993.10 alert=retest1 |

### Cycle 4 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 33410.00 | 34799.57 | 34800.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 13:15:00 | 33260.00 | 34784.25 | 34792.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.89 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-25 10:15:00 | 30590.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-25 11:30:00 | 30555.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-25 12:00:00 | 30590.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-25 15:15:00 | 30630.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-08-29 10:45:00 | 30560.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-08-29 12:00:00 | 30740.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-05 09:15:00 | 30535.00 | 2025-09-05 11:15:00 | 30185.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-08 09:15:00 | 30890.00 | 2025-09-17 12:15:00 | 30360.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-09-11 09:15:00 | 30975.00 | 2025-09-17 12:15:00 | 30360.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-09-16 09:15:00 | 30695.00 | 2025-09-17 12:15:00 | 30360.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-09-17 09:15:00 | 30700.00 | 2025-09-17 13:15:00 | 30195.00 | STOP_HIT | 1.00 | -1.64% |
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
