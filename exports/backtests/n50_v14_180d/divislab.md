# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 6705.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 20
- **Target hits / Stop hits / Partials:** 0 / 20 / 0
- **Avg / median % per leg:** -1.58% / -1.38%
- **Sum % (uncompounded):** -31.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.24% | -16.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.24% | -16.1% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.23% | -15.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.32% | -4.6% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.19% | -11.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.32% | -4.6% |
| retest2 (combined) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.50% | -27.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6412.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6407.30 | 6410.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6241.50 | 6230.99 | 6307.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 09:15:00 | 6138.00 | 6230.75 | 6304.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 10:15:00 | 6149.50 | 6230.42 | 6304.45 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.29 | SL hit (close>ema400) qty=1.00 sl=6269.29 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.29 | SL hit (close>ema400) qty=1.00 sl=6269.29 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 6279.50 | 6189.93 | 6269.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 6241.50 | 6190.44 | 6269.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:15:00 | 6340.00 | 6190.44 | 6269.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 6320.50 | 6191.73 | 6269.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 6231.00 | 6196.78 | 6270.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:45:00 | 6223.00 | 6197.06 | 6270.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 6234.00 | 6216.42 | 6263.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 6237.00 | 6217.71 | 6262.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 6275.00 | 6219.05 | 6262.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 6293.00 | 6219.05 | 6262.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 6304.50 | 6219.90 | 6263.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 6304.50 | 6219.90 | 6263.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 6338.50 | 6221.08 | 6263.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 6349.00 | 6221.08 | 6263.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 6376.00 | 6222.62 | 6264.11 | SL hit (close>static) qty=1.00 sl=6363.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 6376.00 | 6222.62 | 6264.11 | SL hit (close>static) qty=1.00 sl=6363.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 6376.00 | 6222.62 | 6264.11 | SL hit (close>static) qty=1.00 sl=6363.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 6376.00 | 6222.62 | 6264.11 | SL hit (close>static) qty=1.00 sl=6363.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 6342.00 | 6257.85 | 6278.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 6342.00 | 6257.85 | 6278.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 6300.00 | 6264.21 | 6281.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 6300.00 | 6264.21 | 6281.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 6290.00 | 6264.47 | 6281.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 6200.50 | 6277.35 | 6286.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 6304.00 | 6277.15 | 6286.25 | SL hit (close>static) qty=1.00 sl=6302.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6233.00 | 6294.15 | 6294.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6169.00 | 6292.73 | 6293.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 11:15:00 | 6091.00 | 6061.97 | 6149.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 12:00:00 | 6091.00 | 6061.97 | 6149.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6127.50 | 6064.95 | 6145.88 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 6444.50 | 6198.32 | 6198.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 6468.00 | 6205.80 | 6202.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-25 13:00:00 | 6397.50 | 2025-12-08 12:15:00 | 6342.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-02 10:15:00 | 6381.00 | 2025-12-08 12:15:00 | 6342.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-03 10:30:00 | 6382.00 | 2025-12-08 12:15:00 | 6342.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-12-10 10:15:00 | 6381.00 | 2025-12-10 14:15:00 | 6293.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-12-18 10:30:00 | 6427.00 | 2025-12-30 09:15:00 | 6365.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-19 09:15:00 | 6589.00 | 2025-12-30 09:15:00 | 6365.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-12-26 14:45:00 | 6422.50 | 2025-12-30 09:15:00 | 6365.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-30 15:15:00 | 6480.50 | 2026-01-01 09:15:00 | 6355.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-31 14:45:00 | 6406.50 | 2026-01-01 09:15:00 | 6355.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-01-05 09:30:00 | 6407.50 | 2026-01-05 13:15:00 | 6360.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-05 12:45:00 | 6405.00 | 2026-01-05 13:15:00 | 6360.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-06 09:15:00 | 6456.00 | 2026-01-14 12:15:00 | 6373.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-12 14:45:00 | 6494.00 | 2026-01-14 12:15:00 | 6373.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2026-02-04 09:15:00 | 6138.00 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest1 | 2026-02-04 10:15:00 | 6149.50 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-02-12 11:15:00 | 6231.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-02-12 11:45:00 | 6223.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-24 09:30:00 | 6234.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-02-24 13:15:00 | 6237.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-09 09:15:00 | 6200.50 | 2026-03-09 11:15:00 | 6304.00 | STOP_HIT | 1.00 | -1.67% |
