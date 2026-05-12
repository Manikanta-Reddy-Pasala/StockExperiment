# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4600.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 13
- **Target hits / Stop hits / Partials:** 0 / 13 / 0
- **Avg / median % per leg:** -1.54% / -1.46%
- **Sum % (uncompounded):** -20.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.53% | -12.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.53% | -12.2% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.57% | -7.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.57% | -7.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.54% | -20.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 5716.00 | 4939.39 | 4936.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 13:15:00 | 5745.50 | 4947.41 | 4940.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 5994.50 | 6198.11 | 5943.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:30:00 | 6219.50 | 6198.11 | 5943.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 5946.50 | 6195.61 | 5943.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 5936.00 | 6195.61 | 5943.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5695.00 | 6190.63 | 5942.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 5695.00 | 6190.63 | 5942.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 5637.00 | 6185.12 | 5940.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 5637.00 | 6185.12 | 5940.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 5273.00 | 5793.40 | 5795.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 5266.00 | 5788.16 | 5792.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 15:15:00 | 5110.00 | 5108.62 | 5281.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 5091.50 | 5108.62 | 5281.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4519.60 | 4441.85 | 4524.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 4524.00 | 4441.85 | 4524.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 4561.80 | 4443.05 | 4524.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 4555.10 | 4443.05 | 4524.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 4538.80 | 4444.00 | 4524.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 4574.70 | 4444.00 | 4524.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 15:15:00 | 4779.80 | 4584.44 | 4583.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 12:15:00 | 4800.10 | 4591.68 | 4587.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 4463.00 | 4594.67 | 4594.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 4453.80 | 4593.27 | 4594.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 4584.80 | 4559.80 | 4575.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 4686.00 | 4561.05 | 4576.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 4686.00 | 4561.05 | 4576.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4535.00 | 4559.14 | 4574.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 4571.70 | 4559.14 | 4574.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 4594.00 | 4558.61 | 4574.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 4594.00 | 4558.61 | 4574.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 4568.00 | 4558.71 | 4573.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 4541.60 | 4558.71 | 4573.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 4608.00 | 4559.03 | 4574.00 | SL hit (close>static) qty=1.00 sl=4598.80 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 4777.30 | 4584.57 | 4583.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 4818.00 | 4590.77 | 4586.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 4698.80 | 4727.93 | 4670.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 4698.80 | 4727.93 | 4670.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 4665.70 | 4726.42 | 4670.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 4665.70 | 4726.42 | 4670.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 4670.10 | 4725.86 | 4670.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 4699.40 | 4725.38 | 4670.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 4688.00 | 4724.10 | 4670.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 4685.00 | 4723.27 | 4670.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 12:30:00 | 4704.60 | 4722.00 | 4671.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 4614.90 | 4720.94 | 4671.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 4614.90 | 4720.94 | 4671.20 | SL hit (close<static) qty=1.00 sl=4659.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 13:15:00 | 4750.40 | 2025-05-14 09:15:00 | 4808.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-14 14:15:00 | 4841.00 | 2025-05-14 14:15:00 | 4893.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-24 15:15:00 | 4541.60 | 2026-03-25 09:15:00 | 4608.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-25 12:30:00 | 4540.00 | 2026-03-30 14:15:00 | 4641.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-03-30 14:15:00 | 4557.90 | 2026-03-30 14:15:00 | 4641.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-04-27 09:15:00 | 4699.40 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-04-27 12:30:00 | 4688.00 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-04-27 15:15:00 | 4685.00 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-28 12:30:00 | 4704.60 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-05-06 09:15:00 | 4685.80 | 2026-05-06 13:15:00 | 4627.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-05-07 09:15:00 | 4670.00 | 2026-05-08 10:15:00 | 4603.70 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-05-07 09:45:00 | 4668.90 | 2026-05-08 10:15:00 | 4603.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-05-07 13:45:00 | 4669.10 | 2026-05-08 10:15:00 | 4603.70 | STOP_HIT | 1.00 | -1.40% |
