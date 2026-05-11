# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 5391.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 0
- **Avg / median % per leg:** -2.82% / -3.40%
- **Sum % (uncompounded):** -42.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 1 | 9.1% | 1 | 10 | 0 | -1.94% | -21.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.65% | -14.6% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | -0.96% | -6.7% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.25% | -21.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.25% | -21.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.65% | -14.6% |
| retest2 (combined) | 11 | 1 | 9.1% | 1 | 10 | 0 | -2.52% | -27.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 3184.00 | 2932.25 | 2931.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 3274.20 | 2942.77 | 2936.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 3885.20 | 3941.11 | 3792.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:45:00 | 3884.30 | 3941.11 | 3792.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 4313.00 | 4427.16 | 4329.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 4313.00 | 4427.16 | 4329.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 4309.80 | 4426.00 | 4329.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 4309.80 | 4426.00 | 4329.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 4301.20 | 4424.75 | 4329.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 4305.00 | 4424.75 | 4329.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 4017.70 | 4260.57 | 4261.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 4002.10 | 4238.13 | 4249.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:45:00 | 4182.60 | 4143.65 | 4189.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:45:00 | 4178.70 | 4146.18 | 4189.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:45:00 | 4180.00 | 4146.83 | 4189.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 4170.00 | 4147.36 | 4189.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 4317.10 | 4149.49 | 4190.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 4317.10 | 4149.49 | 4190.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 4397.10 | 4155.84 | 4192.82 | SL hit (close>static) qty=1.00 sl=4368.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 4425.60 | 4225.27 | 4224.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 4481.70 | 4231.94 | 4227.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4590.20 | 4603.40 | 4470.54 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:00:00 | 4659.20 | 4603.95 | 4471.48 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 4653.10 | 4611.66 | 4479.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 14:00:00 | 4665.00 | 4613.52 | 4482.86 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 15:15:00 | 4683.90 | 4613.67 | 4483.59 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 4523.30 | 4611.53 | 4486.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:15:00 | 4574.20 | 4611.53 | 4486.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:30:00 | 4559.00 | 4613.95 | 4500.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 4495.00 | 4612.76 | 4500.09 | SL hit (close<ema400) qty=1.00 sl=4500.09 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-03 14:45:00 | 4182.60 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2026-02-04 13:45:00 | 4178.70 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2026-02-04 14:45:00 | 4180.00 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-02-05 09:15:00 | 4170.00 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest1 | 2026-03-12 11:00:00 | 4659.20 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest1 | 2026-03-13 10:45:00 | 4653.10 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest1 | 2026-03-13 14:00:00 | 4665.00 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest1 | 2026-03-13 15:15:00 | 4683.90 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2026-03-16 14:15:00 | 4574.20 | 2026-03-23 09:15:00 | 4452.70 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-03-19 13:30:00 | 4559.00 | 2026-03-23 09:15:00 | 4452.70 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-20 09:15:00 | 4608.70 | 2026-03-23 09:15:00 | 4452.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-03-24 09:15:00 | 4588.10 | 2026-03-30 15:15:00 | 4455.90 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-04-01 09:15:00 | 4655.00 | 2026-04-02 09:15:00 | 4503.40 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-01 12:00:00 | 4606.00 | 2026-04-02 09:15:00 | 4503.40 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-04-02 12:45:00 | 4581.30 | 2026-04-10 12:15:00 | 5039.43 | TARGET_HIT | 1.00 | 10.00% |
