# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 5115.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 10
- **Target hits / Stop hits / Partials:** 0 / 15 / 5
- **Avg / median % per leg:** -0.78% / 0.99%
- **Sum % (uncompounded):** -15.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.24% | -19.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.24% | -19.4% |
| SELL (all) | 14 | 10 | 71.4% | 0 | 9 | 5 | 0.27% | 3.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 0 | 9 | 5 | 0.27% | 3.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 10 | 50.0% | 0 | 15 | 5 | -0.78% | -15.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 5764.50 | 5411.24 | 5410.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 5936.00 | 5551.39 | 5499.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 5873.00 | 5890.37 | 5744.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 5848.50 | 5890.37 | 5744.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5787.00 | 5884.09 | 5753.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 5844.00 | 5754.73 | 5714.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 5808.00 | 5755.59 | 5715.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 5714.00 | 5755.66 | 5716.46 | SL hit (close<static) qty=1.00 sl=5735.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 5174.00 | 5681.19 | 5681.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 5159.00 | 5670.87 | 5676.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 5346.00 | 5342.19 | 5460.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 5346.00 | 5342.19 | 5460.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 5464.00 | 5345.64 | 5451.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 5486.00 | 5345.64 | 5451.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 5494.00 | 5347.11 | 5451.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 5494.00 | 5347.11 | 5451.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 5498.00 | 5348.62 | 5451.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 5477.50 | 5349.97 | 5451.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:15:00 | 5475.00 | 5349.97 | 5451.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 5478.00 | 5351.27 | 5452.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 5203.62 | 5354.26 | 5431.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 5201.25 | 5354.26 | 5431.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 5204.10 | 5354.26 | 5431.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 5369.50 | 5306.81 | 5398.46 | SL hit (close>ema200) qty=0.50 sl=5306.81 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5851.40 | 5368.54 | 5366.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 5918.10 | 5391.71 | 5378.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 6140.50 | 6182.49 | 5952.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 13:00:00 | 6140.50 | 6182.49 | 5952.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 6120.00 | 6317.90 | 6187.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 6105.50 | 6317.90 | 6187.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 6069.50 | 6315.42 | 6187.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 6042.00 | 6315.42 | 6187.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 6149.00 | 6310.77 | 6186.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 6149.00 | 6310.77 | 6186.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 6239.00 | 6310.06 | 6187.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 6364.50 | 6309.16 | 6187.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 13:45:00 | 6260.00 | 6307.02 | 6189.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 6135.00 | 6301.50 | 6190.46 | SL hit (close<static) qty=1.00 sl=6142.50 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5723.00 | 6125.83 | 6127.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 5703.00 | 6102.95 | 6115.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5039.00 | 4963.44 | 5276.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 10:00:00 | 5039.00 | 4963.44 | 5276.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 5233.80 | 4979.72 | 5266.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 5224.20 | 4979.72 | 5266.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 5273.20 | 4985.06 | 5266.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 5117.30 | 5208.24 | 5309.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 4861.44 | 5182.09 | 5287.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 5066.80 | 5033.71 | 5173.12 | SL hit (close>ema200) qty=0.50 sl=5033.71 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-22 09:15:00 | 5844.00 | 2025-07-22 14:15:00 | 5714.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-22 10:30:00 | 5808.00 | 2025-07-22 14:15:00 | 5714.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-25 12:45:00 | 5477.50 | 2025-09-05 09:15:00 | 5203.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 13:15:00 | 5475.00 | 2025-09-05 09:15:00 | 5201.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 13:45:00 | 5478.00 | 2025-09-05 09:15:00 | 5204.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:45:00 | 5477.50 | 2025-09-10 09:15:00 | 5369.50 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-08-25 13:15:00 | 5475.00 | 2025-09-10 09:15:00 | 5369.50 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-08-25 13:45:00 | 5478.00 | 2025-09-10 09:15:00 | 5369.50 | STOP_HIT | 0.50 | 1.98% |
| SELL | retest2 | 2025-09-22 09:15:00 | 5302.50 | 2025-09-26 11:15:00 | 5037.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 5302.50 | 2025-10-06 10:15:00 | 5220.00 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2025-10-13 09:15:00 | 5289.90 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -8.23% |
| SELL | retest2 | 2025-10-14 10:30:00 | 5333.60 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2025-10-14 13:15:00 | 5349.40 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-10-14 15:00:00 | 5348.70 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -7.04% |
| BUY | retest2 | 2026-01-22 09:15:00 | 6364.50 | 2026-01-23 13:15:00 | 6135.00 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-22 13:45:00 | 6260.00 | 2026-01-23 13:15:00 | 6135.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-01-28 09:30:00 | 6278.00 | 2026-01-29 09:15:00 | 6027.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-02-03 09:15:00 | 6299.00 | 2026-02-04 09:15:00 | 5921.50 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2026-04-22 09:15:00 | 5117.30 | 2026-04-24 11:15:00 | 4861.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 5117.30 | 2026-05-08 09:15:00 | 5066.80 | STOP_HIT | 0.50 | 0.99% |
