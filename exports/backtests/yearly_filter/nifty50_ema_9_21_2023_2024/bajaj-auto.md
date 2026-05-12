# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 10696.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 155 |
| ALERT2 | 154 |
| ALERT2_SKIP | 82 |
| ALERT3 | 420 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 194 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 200 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 205 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 143
- **Target hits / Stop hits / Partials:** 7 / 191 / 7
- **Avg / median % per leg:** 0.14% / -0.75%
- **Sum % (uncompounded):** 28.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 90 | 33 | 36.7% | 5 | 84 | 1 | 0.49% | 44.0% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.58% | 10.3% |
| BUY @ 3rd Alert (retest2) | 86 | 30 | 34.9% | 5 | 81 | 0 | 0.39% | 33.7% |
| SELL (all) | 115 | 29 | 25.2% | 2 | 107 | 6 | -0.13% | -15.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.03% | -1.0% |
| SELL @ 3rd Alert (retest2) | 114 | 29 | 25.4% | 2 | 106 | 6 | -0.13% | -14.3% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 1.86% | 9.3% |
| retest2 (combined) | 200 | 59 | 29.5% | 7 | 187 | 6 | 0.10% | 19.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 4536.00 | 4563.24 | 4565.71 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 4585.05 | 4567.60 | 4567.47 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 4544.75 | 4563.03 | 4565.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 4522.35 | 4554.90 | 4561.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 10:15:00 | 4538.90 | 4538.19 | 4548.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 4538.90 | 4538.19 | 4548.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 4538.90 | 4538.19 | 4548.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:30:00 | 4540.00 | 4538.19 | 4548.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 4487.45 | 4518.67 | 4533.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 14:00:00 | 4477.85 | 4500.70 | 4519.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 10:15:00 | 4551.05 | 4514.90 | 4519.90 | SL hit (close>static) qty=1.00 sl=4550.15 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 13:15:00 | 4529.95 | 4523.71 | 4523.23 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 15:15:00 | 4512.00 | 4521.25 | 4522.19 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 4534.10 | 4523.82 | 4523.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 4543.20 | 4527.70 | 4525.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 12:15:00 | 4522.90 | 4527.07 | 4525.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 12:15:00 | 4522.90 | 4527.07 | 4525.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 4522.90 | 4527.07 | 4525.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:00:00 | 4522.90 | 4527.07 | 4525.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 13:15:00 | 4509.65 | 4523.58 | 4523.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 14:15:00 | 4499.85 | 4518.84 | 4521.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 09:15:00 | 4537.85 | 4518.44 | 4520.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 4537.85 | 4518.44 | 4520.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 4537.85 | 4518.44 | 4520.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 10:00:00 | 4537.85 | 4518.44 | 4520.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 4531.95 | 4521.14 | 4521.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 10:45:00 | 4537.00 | 4521.14 | 4521.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 4532.55 | 4523.42 | 4522.74 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 4515.00 | 4521.70 | 4522.21 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 4632.00 | 4543.76 | 4532.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 15:15:00 | 4651.00 | 4608.29 | 4574.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 14:15:00 | 4613.50 | 4627.31 | 4601.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-26 15:00:00 | 4613.50 | 4627.31 | 4601.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 4611.00 | 4625.52 | 4615.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 4628.10 | 4625.52 | 4615.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 12:15:00 | 4602.75 | 4619.50 | 4615.45 | SL hit (close<static) qty=1.00 sl=4608.10 alert=retest2 |

### Cycle 11 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 4590.45 | 4610.64 | 4611.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 13:15:00 | 4569.15 | 4598.96 | 4605.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 4638.05 | 4598.25 | 4602.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 4638.05 | 4598.25 | 4602.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 4638.05 | 4598.25 | 4602.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:00:00 | 4638.05 | 4598.25 | 4602.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 4646.00 | 4607.80 | 4606.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 4649.90 | 4622.76 | 4614.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 10:15:00 | 4783.50 | 4792.26 | 4770.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 10:45:00 | 4781.70 | 4792.26 | 4770.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 4763.00 | 4785.30 | 4770.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:00:00 | 4763.00 | 4785.30 | 4770.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 4764.00 | 4781.04 | 4770.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:30:00 | 4762.70 | 4781.04 | 4770.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 4742.00 | 4761.78 | 4762.97 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 4770.05 | 4763.74 | 4762.92 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 15:15:00 | 4750.55 | 4762.09 | 4763.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 09:15:00 | 4728.95 | 4755.46 | 4760.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 15:15:00 | 4738.70 | 4738.15 | 4747.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-15 09:15:00 | 4746.30 | 4738.15 | 4747.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 4751.35 | 4740.79 | 4748.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 09:45:00 | 4756.00 | 4740.79 | 4748.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 4751.65 | 4742.96 | 4748.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 10:30:00 | 4756.90 | 4742.96 | 4748.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 4734.95 | 4741.36 | 4747.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 12:30:00 | 4727.55 | 4737.62 | 4744.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 14:00:00 | 4723.70 | 4734.83 | 4743.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 09:15:00 | 4689.15 | 4651.48 | 4651.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 4689.15 | 4651.48 | 4651.21 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 4640.00 | 4651.70 | 4652.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 4624.00 | 4646.16 | 4650.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 09:15:00 | 4637.75 | 4631.01 | 4639.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 4637.75 | 4631.01 | 4639.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 4637.75 | 4631.01 | 4639.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:30:00 | 4620.00 | 4631.01 | 4639.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 4628.65 | 4630.54 | 4638.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 11:15:00 | 4619.95 | 4630.54 | 4638.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 13:00:00 | 4620.00 | 4629.95 | 4637.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 15:00:00 | 4621.50 | 4626.67 | 4634.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:00:00 | 4620.00 | 4627.29 | 4632.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 4670.05 | 4624.82 | 4627.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 4670.05 | 4624.82 | 4627.31 | SL hit (close>static) qty=1.00 sl=4655.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 10:15:00 | 4654.90 | 4630.84 | 4629.81 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 14:15:00 | 4618.00 | 4629.38 | 4629.88 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 14:15:00 | 4722.05 | 4647.61 | 4637.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 10:15:00 | 4730.00 | 4683.69 | 4658.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 14:15:00 | 4691.95 | 4694.25 | 4672.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 15:00:00 | 4691.95 | 4694.25 | 4672.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 4637.55 | 4681.91 | 4670.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:00:00 | 4637.55 | 4681.91 | 4670.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 4623.00 | 4670.13 | 4666.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:00:00 | 4623.00 | 4670.13 | 4666.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 11:15:00 | 4607.90 | 4657.68 | 4660.87 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 11:15:00 | 4762.00 | 4667.43 | 4655.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 14:15:00 | 4886.35 | 4738.29 | 4692.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 11:15:00 | 4870.20 | 4875.82 | 4820.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 12:00:00 | 4870.20 | 4875.82 | 4820.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 4825.00 | 4854.56 | 4827.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 09:15:00 | 4909.10 | 4854.56 | 4827.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 4861.45 | 4893.24 | 4895.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 4861.45 | 4893.24 | 4895.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 09:15:00 | 4847.45 | 4863.55 | 4875.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 10:15:00 | 4870.00 | 4864.84 | 4874.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-17 11:00:00 | 4870.00 | 4864.84 | 4874.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 4869.95 | 4865.86 | 4874.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:30:00 | 4868.30 | 4865.86 | 4874.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 4871.85 | 4854.05 | 4863.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 14:15:00 | 4825.00 | 4848.17 | 4855.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 10:45:00 | 4822.05 | 4833.08 | 4845.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 15:15:00 | 4880.00 | 4852.04 | 4850.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 4880.00 | 4852.04 | 4850.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 13:15:00 | 4901.00 | 4875.63 | 4865.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 10:15:00 | 4874.10 | 4878.74 | 4870.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 10:15:00 | 4874.10 | 4878.74 | 4870.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 4874.10 | 4878.74 | 4870.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:00:00 | 4874.10 | 4878.74 | 4870.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 4867.05 | 4876.40 | 4870.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 12:00:00 | 4867.05 | 4876.40 | 4870.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 12:15:00 | 4848.50 | 4870.82 | 4868.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 12:45:00 | 4849.70 | 4870.82 | 4868.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 13:15:00 | 4845.10 | 4865.68 | 4866.03 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 13:15:00 | 4892.85 | 4868.93 | 4866.26 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 14:15:00 | 4858.45 | 4866.53 | 4866.88 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 12:15:00 | 4883.05 | 4867.66 | 4866.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 14:15:00 | 4893.10 | 4873.52 | 4869.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 11:15:00 | 4925.25 | 4927.67 | 4906.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 12:00:00 | 4925.25 | 4927.67 | 4906.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 4922.75 | 4926.69 | 4907.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:30:00 | 4908.70 | 4926.69 | 4907.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 4906.55 | 4922.50 | 4912.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 4901.55 | 4922.50 | 4912.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 4872.70 | 4912.54 | 4908.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 4872.70 | 4912.54 | 4908.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 4864.65 | 4902.96 | 4904.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 12:15:00 | 4843.00 | 4890.97 | 4898.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 4890.70 | 4866.38 | 4882.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 4890.70 | 4866.38 | 4882.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 4890.70 | 4866.38 | 4882.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:30:00 | 4892.40 | 4866.38 | 4882.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 4871.10 | 4867.32 | 4881.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:30:00 | 4877.90 | 4867.32 | 4881.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 4826.55 | 4860.38 | 4875.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 13:15:00 | 4822.85 | 4860.38 | 4875.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 09:15:00 | 4817.50 | 4842.92 | 4862.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 15:15:00 | 4581.71 | 4614.76 | 4636.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 09:15:00 | 4576.62 | 4608.12 | 4631.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-14 13:15:00 | 4606.35 | 4598.76 | 4618.50 | SL hit (close>ema200) qty=0.50 sl=4598.76 alert=retest2 |

### Cycle 30 — BUY (started 2023-08-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 13:15:00 | 4642.30 | 4609.93 | 4606.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 14:15:00 | 4657.25 | 4619.39 | 4611.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 10:15:00 | 4624.45 | 4626.58 | 4617.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-18 11:00:00 | 4624.45 | 4626.58 | 4617.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 4620.75 | 4625.09 | 4618.25 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 14:15:00 | 4606.25 | 4617.02 | 4617.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 15:15:00 | 4604.00 | 4614.42 | 4615.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 11:15:00 | 4621.05 | 4614.61 | 4615.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 11:15:00 | 4621.05 | 4614.61 | 4615.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 11:15:00 | 4621.05 | 4614.61 | 4615.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:00:00 | 4621.05 | 4614.61 | 4615.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 4617.20 | 4615.13 | 4615.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:45:00 | 4621.10 | 4615.13 | 4615.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 13:15:00 | 4620.95 | 4616.29 | 4616.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 15:15:00 | 4630.00 | 4620.57 | 4618.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 4664.70 | 4672.56 | 4654.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 14:00:00 | 4664.70 | 4672.56 | 4654.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 4632.20 | 4664.49 | 4652.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 4632.20 | 4664.49 | 4652.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 4614.00 | 4654.39 | 4648.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 4639.50 | 4654.39 | 4648.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 10:15:00 | 4606.25 | 4637.67 | 4641.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 4606.25 | 4637.67 | 4641.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 4589.95 | 4628.13 | 4636.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 4616.90 | 4608.33 | 4621.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 09:45:00 | 4611.05 | 4608.33 | 4621.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 4642.40 | 4609.55 | 4615.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:45:00 | 4640.40 | 4609.55 | 4615.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 4629.90 | 4613.62 | 4616.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:30:00 | 4647.50 | 4613.62 | 4616.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 4625.00 | 4618.83 | 4618.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 4664.50 | 4629.41 | 4623.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 15:15:00 | 4678.00 | 4678.62 | 4659.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:15:00 | 4682.45 | 4678.62 | 4659.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 4665.00 | 4674.06 | 4660.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:45:00 | 4655.60 | 4674.06 | 4660.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 4655.45 | 4669.86 | 4660.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-31 12:15:00 | 4655.45 | 4669.86 | 4660.79 | SL hit (close<ema400) qty=1.00 sl=4660.79 alert=retest1 |

### Cycle 35 — SELL (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 14:15:00 | 4622.95 | 4654.35 | 4654.93 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 4685.00 | 4660.16 | 4657.01 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 4632.90 | 4658.71 | 4660.33 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 4689.40 | 4657.00 | 4656.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 11:15:00 | 4714.95 | 4668.59 | 4661.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 09:15:00 | 4685.00 | 4686.64 | 4674.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 10:00:00 | 4685.00 | 4686.64 | 4674.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 4672.45 | 4685.16 | 4677.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:00:00 | 4672.45 | 4685.16 | 4677.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 4671.25 | 4682.37 | 4676.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:30:00 | 4671.35 | 4682.37 | 4676.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 4689.75 | 4683.85 | 4677.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 14:30:00 | 4669.80 | 4683.85 | 4677.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 4688.50 | 4684.78 | 4678.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:15:00 | 4665.45 | 4684.78 | 4678.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 4683.20 | 4684.46 | 4679.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:30:00 | 4672.85 | 4684.46 | 4679.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 4686.35 | 4687.95 | 4681.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 11:30:00 | 4684.00 | 4687.95 | 4681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 4678.65 | 4686.09 | 4681.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:45:00 | 4678.05 | 4686.09 | 4681.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 13:15:00 | 4681.90 | 4685.25 | 4681.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 13:30:00 | 4679.15 | 4685.25 | 4681.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 4718.60 | 4691.92 | 4685.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 14:30:00 | 4698.55 | 4691.92 | 4685.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 4807.05 | 4822.44 | 4805.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:45:00 | 4807.65 | 4822.44 | 4805.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 4815.00 | 4820.96 | 4806.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 09:30:00 | 4827.30 | 4823.31 | 4808.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 12:45:00 | 4820.90 | 4820.28 | 4810.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 13:45:00 | 4821.25 | 4822.22 | 4812.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 15:15:00 | 5084.20 | 5119.79 | 5121.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 15:15:00 | 5084.20 | 5119.79 | 5121.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 5062.80 | 5108.39 | 5116.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 5071.00 | 5027.77 | 5047.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 5071.00 | 5027.77 | 5047.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 5071.00 | 5027.77 | 5047.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:00:00 | 5071.00 | 5027.77 | 5047.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 5080.00 | 5038.22 | 5050.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 5081.90 | 5038.22 | 5050.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 13:15:00 | 5083.35 | 5059.44 | 5057.85 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 09:15:00 | 5035.00 | 5057.63 | 5057.71 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 5063.25 | 5058.76 | 5058.22 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 11:15:00 | 5042.30 | 5055.47 | 5056.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 13:15:00 | 5030.00 | 5047.71 | 5052.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 09:15:00 | 5076.95 | 5050.82 | 5052.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 5076.95 | 5050.82 | 5052.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 5076.95 | 5050.82 | 5052.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 5063.45 | 5050.82 | 5052.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 5047.70 | 5050.20 | 5052.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:15:00 | 5035.00 | 5050.20 | 5052.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 13:15:00 | 5080.80 | 5045.09 | 5042.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 5080.80 | 5045.09 | 5042.99 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 5020.05 | 5042.00 | 5043.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 4980.00 | 5026.88 | 5036.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 4954.95 | 4952.36 | 4985.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 10:00:00 | 4954.95 | 4952.36 | 4985.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 4974.50 | 4955.82 | 4976.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:00:00 | 4974.50 | 4955.82 | 4976.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 4994.30 | 4963.52 | 4978.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 15:00:00 | 4994.30 | 4963.52 | 4978.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 5024.55 | 4975.73 | 4982.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:15:00 | 5038.00 | 4975.73 | 4982.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 5039.00 | 4997.43 | 4991.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 09:15:00 | 5055.90 | 5019.61 | 5012.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-10 13:15:00 | 5024.95 | 5028.95 | 5019.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-10 14:00:00 | 5024.95 | 5028.95 | 5019.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 5148.85 | 5076.24 | 5052.76 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 5050.00 | 5069.29 | 5071.89 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 15:15:00 | 5079.80 | 5073.27 | 5072.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 5098.25 | 5078.26 | 5075.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 14:15:00 | 5127.95 | 5151.31 | 5128.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 5127.95 | 5151.31 | 5128.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 5127.95 | 5151.31 | 5128.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 5127.95 | 5151.31 | 5128.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 5143.70 | 5149.79 | 5129.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 5348.25 | 5149.79 | 5129.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-25 14:15:00 | 5349.95 | 5384.21 | 5386.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 14:15:00 | 5349.95 | 5384.21 | 5386.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 15:15:00 | 5342.50 | 5375.87 | 5382.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 5320.20 | 5298.75 | 5329.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 5320.20 | 5298.75 | 5329.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 5320.20 | 5298.75 | 5329.58 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 5399.00 | 5354.85 | 5349.48 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 11:15:00 | 5323.40 | 5344.70 | 5347.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 14:15:00 | 5308.00 | 5334.80 | 5341.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 10:15:00 | 5342.95 | 5335.06 | 5340.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 10:15:00 | 5342.95 | 5335.06 | 5340.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 5342.95 | 5335.06 | 5340.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:00:00 | 5342.95 | 5335.06 | 5340.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 5322.40 | 5332.53 | 5338.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:45:00 | 5340.00 | 5332.53 | 5338.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 5375.00 | 5332.98 | 5335.05 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 10:15:00 | 5370.00 | 5340.39 | 5338.23 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 11:15:00 | 5296.65 | 5334.59 | 5339.27 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 12:15:00 | 5365.00 | 5340.88 | 5338.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 5413.50 | 5366.62 | 5352.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 13:15:00 | 5412.60 | 5413.86 | 5394.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 14:00:00 | 5412.60 | 5413.86 | 5394.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 5424.00 | 5416.60 | 5400.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 10:30:00 | 5450.80 | 5418.08 | 5408.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 09:15:00 | 5360.15 | 5411.83 | 5411.74 | SL hit (close<static) qty=1.00 sl=5381.25 alert=retest2 |

### Cycle 55 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 5369.90 | 5403.44 | 5407.93 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 5428.60 | 5407.31 | 5406.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 5465.30 | 5427.32 | 5417.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 15:15:00 | 5633.65 | 5635.04 | 5604.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:15:00 | 5698.35 | 5635.04 | 5604.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 14:15:00 | 5983.27 | 5944.94 | 5899.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-11-30 09:15:00 | 6032.00 | 6032.87 | 5983.67 | SL hit (close<ema200) qty=0.50 sl=6032.87 alert=retest1 |

### Cycle 57 — SELL (started 2023-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 10:15:00 | 6026.50 | 6099.24 | 6101.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 6005.00 | 6067.89 | 6085.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 14:15:00 | 6072.65 | 6063.90 | 6080.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 14:15:00 | 6072.65 | 6063.90 | 6080.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 6072.65 | 6063.90 | 6080.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 14:45:00 | 6078.95 | 6063.90 | 6080.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 6077.40 | 6066.60 | 6080.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:15:00 | 6089.90 | 6066.60 | 6080.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 6106.70 | 6074.62 | 6082.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:00:00 | 6106.70 | 6074.62 | 6082.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 6122.80 | 6084.26 | 6086.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 11:00:00 | 6122.80 | 6084.26 | 6086.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 11:15:00 | 6128.35 | 6093.07 | 6090.19 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 6082.60 | 6092.18 | 6092.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 6064.40 | 6086.63 | 6090.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 10:15:00 | 6084.25 | 6080.54 | 6085.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 10:15:00 | 6084.25 | 6080.54 | 6085.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 6084.25 | 6080.54 | 6085.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 11:00:00 | 6084.25 | 6080.54 | 6085.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 11:15:00 | 6101.55 | 6084.74 | 6086.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 11:30:00 | 6121.95 | 6084.74 | 6086.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 12:15:00 | 6115.15 | 6090.82 | 6089.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 6134.00 | 6099.46 | 6093.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 11:15:00 | 6297.90 | 6300.46 | 6254.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 12:00:00 | 6297.90 | 6300.46 | 6254.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 6291.75 | 6322.58 | 6294.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:00:00 | 6291.75 | 6322.58 | 6294.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 6286.05 | 6315.27 | 6293.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:45:00 | 6295.40 | 6315.27 | 6293.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 6272.25 | 6306.67 | 6291.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 6272.25 | 6306.67 | 6291.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 6275.95 | 6300.52 | 6290.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 09:15:00 | 6403.50 | 6300.52 | 6290.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-21 09:15:00 | 6284.00 | 6377.86 | 6389.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 6284.00 | 6377.86 | 6389.52 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 6394.35 | 6344.28 | 6342.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 6440.00 | 6363.42 | 6351.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 6668.20 | 6681.33 | 6612.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 10:00:00 | 6668.20 | 6681.33 | 6612.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 6695.45 | 6739.78 | 6697.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:00:00 | 6695.45 | 6739.78 | 6697.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 6703.10 | 6732.44 | 6697.57 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 6581.20 | 6678.18 | 6681.56 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 14:15:00 | 6961.00 | 6706.62 | 6682.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 7100.55 | 7013.29 | 6960.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 10:15:00 | 7056.50 | 7080.19 | 7033.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 10:15:00 | 7056.50 | 7080.19 | 7033.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 7056.50 | 7080.19 | 7033.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 7046.80 | 7080.19 | 7033.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 13:15:00 | 7061.00 | 7067.50 | 7038.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 09:15:00 | 7102.70 | 7061.13 | 7040.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 7105.65 | 7261.77 | 7268.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 7105.65 | 7261.77 | 7268.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 14:15:00 | 7078.50 | 7126.03 | 7173.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 7165.00 | 7127.97 | 7165.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 7165.00 | 7127.97 | 7165.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 7165.00 | 7127.97 | 7165.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:00:00 | 7165.00 | 7127.97 | 7165.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 7135.10 | 7129.40 | 7162.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:45:00 | 7163.00 | 7129.40 | 7162.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 7135.00 | 7130.52 | 7160.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:30:00 | 7160.90 | 7130.52 | 7160.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 7143.45 | 7135.85 | 7151.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 15:15:00 | 7084.00 | 7140.06 | 7148.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 7191.45 | 7141.37 | 7147.13 | SL hit (close>static) qty=1.00 sl=7189.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 13:15:00 | 7181.00 | 7142.64 | 7138.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 14:15:00 | 7216.65 | 7157.44 | 7145.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 7568.95 | 7576.25 | 7511.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 7568.95 | 7576.25 | 7511.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 7611.95 | 7574.47 | 7530.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 12:15:00 | 7633.20 | 7574.47 | 7530.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 11:15:00 | 7687.70 | 7704.60 | 7706.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 11:15:00 | 7687.70 | 7704.60 | 7706.25 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 7736.60 | 7710.90 | 7708.28 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 7680.25 | 7704.77 | 7705.74 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 11:15:00 | 7743.00 | 7712.42 | 7709.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 12:15:00 | 7771.25 | 7724.18 | 7714.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 7685.20 | 7729.77 | 7722.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 10:15:00 | 7685.20 | 7729.77 | 7722.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 7685.20 | 7729.77 | 7722.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 7685.20 | 7729.77 | 7722.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 7747.00 | 7733.21 | 7724.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:15:00 | 7753.90 | 7733.21 | 7724.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 13:45:00 | 7759.00 | 7743.04 | 7730.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-19 09:15:00 | 8529.29 | 8345.42 | 8217.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 8291.00 | 8321.33 | 8322.48 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 12:15:00 | 8331.80 | 8323.42 | 8323.32 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 8306.40 | 8320.02 | 8321.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 8225.55 | 8301.12 | 8313.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 8325.00 | 8286.62 | 8300.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 11:15:00 | 8325.00 | 8286.62 | 8300.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 8325.00 | 8286.62 | 8300.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 8325.00 | 8286.62 | 8300.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 8319.90 | 8293.28 | 8301.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:00:00 | 8319.90 | 8293.28 | 8301.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 8367.35 | 8308.09 | 8307.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 14:15:00 | 8489.30 | 8344.33 | 8324.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 8445.05 | 8465.17 | 8430.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 14:45:00 | 8450.00 | 8465.17 | 8430.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 8428.95 | 8455.50 | 8431.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 8428.95 | 8455.50 | 8431.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 8423.50 | 8449.10 | 8431.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:45:00 | 8411.00 | 8449.10 | 8431.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 8448.45 | 8448.97 | 8432.69 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 8220.00 | 8400.47 | 8418.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 8171.00 | 8354.58 | 8395.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 8048.90 | 8015.57 | 8134.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-01 09:30:00 | 8056.05 | 8015.57 | 8134.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 8045.00 | 8040.90 | 8087.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 8035.00 | 8040.90 | 8087.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 8147.95 | 8062.96 | 8089.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:30:00 | 8144.40 | 8062.96 | 8089.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 8200.00 | 8090.37 | 8099.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:30:00 | 8197.25 | 8090.37 | 8099.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 11:15:00 | 8210.10 | 8114.32 | 8109.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 09:15:00 | 8368.00 | 8209.44 | 8161.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 8719.00 | 8782.02 | 8631.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 10:00:00 | 8719.00 | 8782.02 | 8631.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 8649.65 | 8719.71 | 8654.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 15:00:00 | 8649.65 | 8719.71 | 8654.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 8654.20 | 8706.61 | 8654.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:15:00 | 8625.85 | 8706.61 | 8654.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 8621.45 | 8689.58 | 8651.76 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 12:15:00 | 8525.00 | 8616.07 | 8623.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 8499.40 | 8556.89 | 8590.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 14:15:00 | 8388.15 | 8383.98 | 8443.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 15:00:00 | 8388.15 | 8383.98 | 8443.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 8331.50 | 8374.45 | 8428.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:30:00 | 8223.40 | 8323.71 | 8400.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 11:15:00 | 8515.00 | 8370.29 | 8373.04 | SL hit (close>static) qty=1.00 sl=8432.10 alert=retest2 |

### Cycle 78 — BUY (started 2024-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 12:15:00 | 8515.25 | 8399.28 | 8385.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 09:15:00 | 8636.50 | 8494.48 | 8439.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 14:15:00 | 8639.50 | 8653.60 | 8588.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-20 15:00:00 | 8639.50 | 8653.60 | 8588.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 8625.00 | 8647.88 | 8592.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:45:00 | 8664.90 | 8656.98 | 8601.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 15:15:00 | 9084.45 | 9138.87 | 9145.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 15:15:00 | 9084.45 | 9138.87 | 9145.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 8990.00 | 9109.10 | 9131.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 12:15:00 | 9087.80 | 9086.81 | 9114.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 12:15:00 | 9087.80 | 9086.81 | 9114.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 9087.80 | 9086.81 | 9114.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:45:00 | 9097.95 | 9086.81 | 9114.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 9150.00 | 9099.45 | 9117.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:00:00 | 9150.00 | 9099.45 | 9117.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 9135.00 | 9106.56 | 9119.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:45:00 | 9156.50 | 9106.56 | 9119.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 9064.00 | 9037.13 | 9064.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 11:00:00 | 9064.00 | 9037.13 | 9064.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 11:15:00 | 9004.00 | 9030.51 | 9059.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 13:15:00 | 8989.50 | 9026.43 | 9041.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 09:45:00 | 8945.95 | 8984.19 | 9014.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 14:45:00 | 8987.45 | 8966.19 | 8991.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 10:15:00 | 9099.25 | 9013.27 | 9007.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 9099.25 | 9013.27 | 9007.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 11:15:00 | 9112.20 | 9067.81 | 9044.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 12:15:00 | 9017.10 | 9057.67 | 9042.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 12:15:00 | 9017.10 | 9057.67 | 9042.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 9017.10 | 9057.67 | 9042.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:00:00 | 9017.10 | 9057.67 | 9042.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 9001.25 | 9046.38 | 9038.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 14:00:00 | 9001.25 | 9046.38 | 9038.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 15:15:00 | 9006.45 | 9029.80 | 9031.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 09:15:00 | 8960.25 | 9015.89 | 9025.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 14:15:00 | 8937.85 | 8933.67 | 8975.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 15:00:00 | 8937.85 | 8933.67 | 8975.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 9042.00 | 8952.11 | 8976.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 9030.70 | 8952.11 | 8976.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 9027.00 | 8967.09 | 8981.01 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 9103.10 | 8994.29 | 8992.11 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 8812.75 | 8988.42 | 8995.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 8720.00 | 8934.74 | 8970.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 13:15:00 | 8813.60 | 8796.62 | 8828.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 13:15:00 | 8813.60 | 8796.62 | 8828.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 8813.60 | 8796.62 | 8828.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:00:00 | 8813.60 | 8796.62 | 8828.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 8839.00 | 8804.26 | 8824.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 8861.00 | 8804.26 | 8824.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 8831.25 | 8809.66 | 8824.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:30:00 | 8844.95 | 8809.66 | 8824.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 8795.20 | 8806.77 | 8822.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 13:00:00 | 8775.05 | 8800.42 | 8817.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 14:30:00 | 8759.90 | 8776.67 | 8803.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 8890.60 | 8762.73 | 8770.41 | SL hit (close>static) qty=1.00 sl=8834.65 alert=retest2 |

### Cycle 84 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 8904.50 | 8791.09 | 8782.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 13:15:00 | 8969.10 | 8868.60 | 8823.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 8846.75 | 8891.89 | 8848.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 8846.75 | 8891.89 | 8848.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 8846.75 | 8891.89 | 8848.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 8846.75 | 8891.89 | 8848.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 8820.90 | 8877.69 | 8845.77 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 14:15:00 | 8766.45 | 8819.27 | 8825.29 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 8905.00 | 8828.53 | 8827.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 11:15:00 | 8955.05 | 8868.87 | 8847.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 13:15:00 | 9126.40 | 9129.03 | 9051.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 13:45:00 | 9123.90 | 9129.03 | 9051.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 8837.10 | 9061.82 | 9039.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:45:00 | 8832.65 | 9061.82 | 9039.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 8940.40 | 9037.53 | 9030.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:15:00 | 8980.85 | 9037.53 | 9030.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 8935.80 | 9023.35 | 9028.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 8935.80 | 9023.35 | 9028.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 8827.55 | 8984.19 | 9009.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 15:15:00 | 8751.00 | 8750.57 | 8817.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-09 09:15:00 | 8940.15 | 8750.57 | 8817.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 8976.40 | 8795.74 | 8832.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:45:00 | 8987.30 | 8795.74 | 8832.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 8893.20 | 8815.23 | 8837.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:30:00 | 8822.05 | 8828.44 | 8840.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:15:00 | 8833.80 | 8837.64 | 8843.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 15:00:00 | 8837.50 | 8837.61 | 8843.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 8973.60 | 8865.20 | 8854.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 09:15:00 | 8973.60 | 8865.20 | 8854.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 9017.90 | 8976.25 | 8944.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 8967.00 | 9009.95 | 8976.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 8967.00 | 9009.95 | 8976.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 8967.00 | 9009.95 | 8976.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 8967.00 | 9009.95 | 8976.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 8965.00 | 9000.96 | 8975.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 8950.05 | 9000.96 | 8975.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 8944.50 | 8989.67 | 8972.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:45:00 | 8944.95 | 8989.67 | 8972.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 8921.75 | 8976.08 | 8968.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 8905.75 | 8976.08 | 8968.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 8915.00 | 8956.89 | 8960.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 8892.00 | 8943.91 | 8954.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 8878.60 | 8837.03 | 8884.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 8878.60 | 8837.03 | 8884.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 8878.60 | 8837.03 | 8884.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 8878.60 | 8837.03 | 8884.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 8874.00 | 8844.42 | 8883.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 8905.00 | 8844.42 | 8883.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 8844.95 | 8844.53 | 8880.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:45:00 | 8810.00 | 8839.36 | 8874.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-18 12:15:00 | 8808.75 | 8807.63 | 8836.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 14:30:00 | 8816.55 | 8821.49 | 8829.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 15:15:00 | 8800.00 | 8821.49 | 8829.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 8800.00 | 8817.19 | 8827.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 8787.90 | 8817.19 | 8827.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 8791.15 | 8811.99 | 8823.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 12:30:00 | 8752.70 | 8801.71 | 8816.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 8888.80 | 8824.03 | 8819.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 8888.80 | 8824.03 | 8819.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 8946.40 | 8848.51 | 8831.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 10:15:00 | 8895.45 | 8896.44 | 8863.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:45:00 | 8918.25 | 8896.44 | 8863.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 8925.65 | 8937.33 | 8906.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 8984.50 | 8951.85 | 8915.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 11:00:00 | 9008.00 | 8985.78 | 8953.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 8988.50 | 8980.66 | 8953.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 13:00:00 | 8980.00 | 8980.53 | 8956.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 8995.50 | 8983.53 | 8959.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 8995.50 | 8983.53 | 8959.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 8938.95 | 8975.18 | 8961.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:45:00 | 8983.00 | 8969.54 | 8961.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 8999.15 | 9014.99 | 9001.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 12:45:00 | 8987.95 | 8998.86 | 8997.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 8911.90 | 9185.69 | 9163.24 | SL hit (close<static) qty=1.00 sl=8915.05 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 13:15:00 | 9064.00 | 9148.97 | 9149.68 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 9280.40 | 9175.25 | 9161.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 9546.50 | 9269.46 | 9208.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 9855.40 | 9872.84 | 9818.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 9861.95 | 9872.84 | 9818.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 9849.75 | 9881.21 | 9849.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:15:00 | 9898.50 | 9881.21 | 9849.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 9897.85 | 9917.41 | 9911.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 9779.85 | 9889.90 | 9899.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 9779.85 | 9889.90 | 9899.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 9749.20 | 9841.54 | 9874.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 9682.80 | 9660.52 | 9725.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 9682.80 | 9660.52 | 9725.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 9697.50 | 9667.92 | 9722.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 9682.00 | 9667.92 | 9722.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 9651.35 | 9640.15 | 9678.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 9658.40 | 9640.15 | 9678.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 9715.00 | 9655.12 | 9681.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 9734.00 | 9655.12 | 9681.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 9704.25 | 9664.95 | 9683.86 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 9752.95 | 9704.75 | 9699.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 9768.00 | 9717.40 | 9705.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 9706.60 | 9716.72 | 9707.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 11:15:00 | 9706.60 | 9716.72 | 9707.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 9706.60 | 9716.72 | 9707.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:00:00 | 9706.60 | 9716.72 | 9707.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 9721.85 | 9717.75 | 9708.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:30:00 | 9697.30 | 9717.75 | 9708.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 9665.00 | 9707.20 | 9704.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 9665.00 | 9707.20 | 9704.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 9656.90 | 9697.14 | 9700.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 15:15:00 | 9650.00 | 9687.71 | 9695.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 9467.15 | 9444.49 | 9509.58 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:00:00 | 9403.30 | 9436.25 | 9499.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 9500.00 | 9457.00 | 9490.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 9500.00 | 9457.00 | 9490.25 | SL hit (close>ema400) qty=1.00 sl=9490.25 alert=retest1 |

### Cycle 96 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 9607.25 | 9521.53 | 9512.34 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 9496.85 | 9520.49 | 9520.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 9431.75 | 9502.74 | 9512.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 9530.25 | 9460.80 | 9482.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 9530.25 | 9460.80 | 9482.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 9530.25 | 9460.80 | 9482.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 9530.25 | 9460.80 | 9482.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 9483.55 | 9465.35 | 9482.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 9507.65 | 9465.35 | 9482.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 9418.00 | 9455.88 | 9476.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 9407.35 | 9444.96 | 9469.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 9530.00 | 9474.62 | 9467.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 9530.00 | 9474.62 | 9467.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 9639.50 | 9539.82 | 9504.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 13:15:00 | 9499.70 | 9557.28 | 9533.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 13:15:00 | 9499.70 | 9557.28 | 9533.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 9499.70 | 9557.28 | 9533.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 9499.70 | 9557.28 | 9533.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 9530.00 | 9551.82 | 9533.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 9620.00 | 9533.88 | 9529.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 9478.45 | 9527.81 | 9528.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 9478.45 | 9527.81 | 9528.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 09:15:00 | 9448.05 | 9507.02 | 9516.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 9470.40 | 9464.24 | 9487.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 14:30:00 | 9467.45 | 9464.24 | 9487.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 9495.00 | 9470.39 | 9488.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 9532.00 | 9470.39 | 9488.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 9529.95 | 9482.30 | 9492.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 9543.95 | 9482.30 | 9492.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 9516.15 | 9489.07 | 9494.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 9443.55 | 9483.21 | 9490.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 13:30:00 | 9453.35 | 9476.18 | 9487.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 9590.00 | 9486.63 | 9487.92 | SL hit (close>static) qty=1.00 sl=9532.45 alert=retest2 |

### Cycle 100 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 9575.00 | 9504.30 | 9495.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 9692.50 | 9541.94 | 9513.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 15:15:00 | 9693.00 | 9696.41 | 9637.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 09:15:00 | 9443.95 | 9696.41 | 9637.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 9490.50 | 9655.23 | 9624.39 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 9362.15 | 9596.62 | 9600.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 9342.00 | 9461.34 | 9518.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 9420.00 | 9416.77 | 9461.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 9412.15 | 9416.77 | 9461.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 9473.00 | 9424.28 | 9457.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 9473.00 | 9424.28 | 9457.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 9471.60 | 9433.75 | 9458.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:30:00 | 9497.40 | 9433.75 | 9458.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 9304.95 | 9407.99 | 9444.61 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 9465.30 | 9342.14 | 9338.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 9490.00 | 9417.26 | 9379.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 9675.40 | 9693.04 | 9648.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 9675.40 | 9693.04 | 9648.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 9675.40 | 9693.04 | 9648.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:45:00 | 9717.40 | 9693.96 | 9652.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 9445.15 | 9615.28 | 9631.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 9445.15 | 9615.28 | 9631.27 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 9691.25 | 9564.20 | 9552.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 9708.00 | 9592.96 | 9567.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 9636.65 | 9645.11 | 9604.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 10:00:00 | 9636.65 | 9645.11 | 9604.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 9636.20 | 9666.97 | 9633.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 9636.20 | 9666.97 | 9633.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 9644.45 | 9662.47 | 9634.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 9752.95 | 9662.47 | 9634.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 12:30:00 | 9665.70 | 9673.08 | 9649.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:30:00 | 9673.95 | 9699.11 | 9675.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-28 11:15:00 | 10632.27 | 10534.39 | 10425.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 10914.00 | 10956.27 | 10960.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 12:15:00 | 10854.70 | 10935.96 | 10950.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 10844.70 | 10824.86 | 10852.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 10844.70 | 10824.86 | 10852.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 10853.00 | 10830.49 | 10852.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 10850.75 | 10829.42 | 10849.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 10846.15 | 10832.77 | 10849.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:30:00 | 10835.60 | 10832.77 | 10849.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 10922.90 | 10850.79 | 10856.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:00:00 | 10922.90 | 10850.79 | 10856.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 10931.85 | 10867.00 | 10862.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 10990.15 | 10891.63 | 10874.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 11705.10 | 11724.53 | 11594.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:15:00 | 11779.30 | 11699.90 | 11640.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 11786.20 | 11871.90 | 11805.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 11786.20 | 11871.90 | 11805.80 | SL hit (close<ema400) qty=1.00 sl=11805.80 alert=retest1 |

### Cycle 107 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 12367.75 | 12491.80 | 12497.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 12316.25 | 12456.69 | 12480.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 11914.90 | 11865.89 | 12018.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 11914.90 | 11865.89 | 12018.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 11771.70 | 11679.11 | 11762.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 11758.50 | 11679.11 | 11762.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 11837.70 | 11710.83 | 11769.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 11837.70 | 11710.83 | 11769.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 11882.30 | 11745.13 | 11779.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:30:00 | 11825.55 | 11774.63 | 11790.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 11913.50 | 11802.40 | 11801.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 11913.50 | 11802.40 | 11801.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 11991.05 | 11871.76 | 11843.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 11864.65 | 11880.01 | 11855.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 11864.65 | 11880.01 | 11855.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 11864.65 | 11880.01 | 11855.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 11864.65 | 11880.01 | 11855.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 11854.65 | 11874.93 | 11855.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 11854.65 | 11874.93 | 11855.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 11834.40 | 11866.83 | 11853.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 11823.10 | 11866.83 | 11853.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 11860.00 | 11865.46 | 11854.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 11922.75 | 11865.46 | 11854.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 11766.50 | 11845.67 | 11846.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 11766.50 | 11845.67 | 11846.07 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 15:15:00 | 11869.50 | 11843.14 | 11841.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 11940.05 | 11862.52 | 11850.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 15:15:00 | 11880.30 | 11889.02 | 11871.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 09:15:00 | 11823.15 | 11889.02 | 11871.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 111 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 11679.50 | 11847.12 | 11854.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 10:15:00 | 11585.55 | 11794.80 | 11829.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 11:15:00 | 11627.90 | 11581.25 | 11671.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 12:00:00 | 11627.90 | 11581.25 | 11671.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 11619.40 | 11594.72 | 11655.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:30:00 | 11662.95 | 11594.72 | 11655.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 10400.00 | 10197.41 | 10475.95 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 10645.70 | 10476.10 | 10472.31 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 10275.00 | 10483.34 | 10492.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 10037.95 | 10295.68 | 10384.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 10205.85 | 10154.04 | 10265.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 15:00:00 | 10205.85 | 10154.04 | 10265.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 10150.70 | 10161.77 | 10250.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:30:00 | 10074.00 | 10120.18 | 10197.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 09:15:00 | 9570.30 | 9803.31 | 9876.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 9645.00 | 9581.45 | 9694.39 | SL hit (close>ema200) qty=0.50 sl=9581.45 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 9845.00 | 9740.78 | 9734.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 10000.95 | 9792.81 | 9758.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 9828.75 | 9927.82 | 9864.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 9828.75 | 9927.82 | 9864.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 9828.75 | 9927.82 | 9864.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 9828.75 | 9927.82 | 9864.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 9858.45 | 9913.95 | 9863.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 9829.25 | 9913.95 | 9863.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 9870.40 | 9905.24 | 9864.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 9898.30 | 9905.24 | 9864.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 13:45:00 | 9896.00 | 9898.14 | 9867.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 9915.20 | 9881.37 | 9865.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 10:30:00 | 9893.00 | 9888.22 | 9871.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 9887.50 | 9888.08 | 9872.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 9887.50 | 9888.08 | 9872.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 9905.00 | 9891.46 | 9875.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:15:00 | 9925.00 | 9891.46 | 9875.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:30:00 | 9914.20 | 9896.02 | 9880.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:30:00 | 9963.60 | 9914.38 | 9891.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 13:15:00 | 9858.95 | 9922.98 | 9905.98 | SL hit (close<static) qty=1.00 sl=9870.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 9820.00 | 9888.76 | 9894.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 9772.10 | 9865.43 | 9883.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 13:15:00 | 9499.35 | 9496.73 | 9586.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 9559.55 | 9511.10 | 9570.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 9559.55 | 9511.10 | 9570.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 9559.55 | 9511.10 | 9570.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 9530.85 | 9515.05 | 9566.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:15:00 | 9577.70 | 9515.05 | 9566.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 9567.70 | 9525.58 | 9567.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:45:00 | 9576.10 | 9525.58 | 9567.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 9574.35 | 9535.33 | 9567.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:45:00 | 9572.75 | 9535.33 | 9567.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 9545.90 | 9537.45 | 9565.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:30:00 | 9555.75 | 9537.45 | 9565.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 9580.00 | 9539.32 | 9558.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 9530.10 | 9539.32 | 9558.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 9615.00 | 9554.46 | 9563.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 9615.00 | 9554.46 | 9563.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 9644.05 | 9572.38 | 9571.26 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 9512.05 | 9577.06 | 9577.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 11:15:00 | 9500.30 | 9545.04 | 9561.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 9526.55 | 9518.53 | 9538.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 9526.55 | 9518.53 | 9538.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 9526.55 | 9518.53 | 9538.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 15:00:00 | 9400.05 | 9496.69 | 9517.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 10:15:00 | 9206.00 | 9135.75 | 9126.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 9206.00 | 9135.75 | 9126.80 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 8968.15 | 9127.82 | 9133.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 8833.30 | 9020.16 | 9076.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 14:15:00 | 8915.00 | 8906.70 | 8988.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 15:00:00 | 8915.00 | 8906.70 | 8988.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 9057.05 | 8936.50 | 8987.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 9057.05 | 8936.50 | 8987.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 9108.95 | 8970.99 | 8998.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 9108.95 | 8970.99 | 8998.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 9101.85 | 9031.09 | 9022.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 9120.00 | 9069.70 | 9045.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 13:15:00 | 9086.35 | 9086.84 | 9060.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 13:45:00 | 9089.25 | 9086.84 | 9060.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 9072.95 | 9080.09 | 9061.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 8993.20 | 9080.09 | 9061.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 9035.10 | 9071.09 | 9059.41 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 9014.95 | 9051.53 | 9052.03 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 9082.80 | 9051.59 | 9048.28 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 8998.05 | 9041.84 | 9046.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 13:15:00 | 8973.65 | 9018.00 | 9033.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 9007.70 | 8983.76 | 9007.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 9007.70 | 8983.76 | 9007.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 9007.70 | 8983.76 | 9007.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 9007.70 | 8983.76 | 9007.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 9013.15 | 8989.64 | 9008.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 9013.15 | 8989.64 | 9008.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 9040.00 | 8999.71 | 9011.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 9040.00 | 8999.71 | 9011.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 9021.65 | 9004.10 | 9011.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 9023.35 | 9004.10 | 9011.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 8977.45 | 9006.04 | 9011.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 8970.10 | 8993.97 | 9002.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:00:00 | 8973.95 | 8984.89 | 8996.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:45:00 | 8959.95 | 8930.38 | 8956.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:30:00 | 8974.15 | 8936.23 | 8957.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 8957.00 | 8940.38 | 8957.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:15:00 | 8961.25 | 8940.38 | 8957.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 8970.75 | 8946.45 | 8958.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:00:00 | 8970.75 | 8946.45 | 8958.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 8954.00 | 8947.96 | 8957.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 8836.30 | 8949.37 | 8957.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 8981.20 | 8929.70 | 8941.00 | SL hit (close>static) qty=1.00 sl=8978.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 15:15:00 | 8980.00 | 8948.17 | 8947.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 9000.00 | 8958.54 | 8952.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 8908.85 | 8962.02 | 8957.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 8908.85 | 8962.02 | 8957.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 8908.85 | 8962.02 | 8957.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:45:00 | 8925.60 | 8962.02 | 8957.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 8841.10 | 8937.84 | 8946.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 8771.25 | 8904.52 | 8930.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 8812.80 | 8796.73 | 8844.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 8812.80 | 8796.73 | 8844.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 8812.80 | 8796.73 | 8844.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 8812.80 | 8796.73 | 8844.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 8833.80 | 8793.95 | 8817.35 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 8865.00 | 8831.41 | 8829.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 9154.30 | 8907.65 | 8866.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 8945.70 | 8964.85 | 8916.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 14:15:00 | 8945.70 | 8964.85 | 8916.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 8945.70 | 8964.85 | 8916.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 8945.70 | 8964.85 | 8916.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 8825.00 | 8934.98 | 8911.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 8825.00 | 8934.98 | 8911.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 8869.95 | 8921.98 | 8907.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 8885.00 | 8908.81 | 8902.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 12:15:00 | 8818.00 | 8890.65 | 8895.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 8818.00 | 8890.65 | 8895.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 8756.00 | 8852.51 | 8876.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 8821.75 | 8819.38 | 8846.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 8821.75 | 8819.38 | 8846.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 8807.00 | 8737.86 | 8768.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 8807.00 | 8737.86 | 8768.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 8828.95 | 8756.08 | 8773.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:45:00 | 8811.45 | 8756.08 | 8773.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 8904.10 | 8804.34 | 8793.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 9083.00 | 8860.07 | 8820.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 8965.50 | 8984.76 | 8925.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 8965.50 | 8984.76 | 8925.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 8893.00 | 8967.46 | 8931.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 8893.00 | 8967.46 | 8931.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 8905.00 | 8954.97 | 8929.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 8840.50 | 8954.97 | 8929.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 8812.05 | 8902.48 | 8909.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 8746.30 | 8815.03 | 8851.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 8797.05 | 8722.66 | 8774.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 8797.05 | 8722.66 | 8774.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 8797.05 | 8722.66 | 8774.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:45:00 | 8664.95 | 8777.98 | 8784.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 8672.55 | 8769.70 | 8775.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 10:45:00 | 8696.60 | 8723.16 | 8751.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:45:00 | 8660.00 | 8611.83 | 8645.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 8623.75 | 8614.22 | 8643.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 8645.90 | 8614.22 | 8643.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 8605.05 | 8612.38 | 8639.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:30:00 | 8637.70 | 8612.38 | 8639.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 8594.40 | 8608.79 | 8635.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:30:00 | 8622.25 | 8608.79 | 8635.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 8590.45 | 8591.69 | 8617.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 8623.35 | 8591.69 | 8617.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 8599.20 | 8586.59 | 8610.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 8599.20 | 8586.59 | 8610.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 8573.90 | 8584.05 | 8607.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:45:00 | 8607.90 | 8584.05 | 8607.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 8625.55 | 8595.10 | 8605.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 8605.05 | 8595.10 | 8605.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 8572.55 | 8590.59 | 8602.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 8630.40 | 8590.59 | 8602.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 8595.00 | 8591.47 | 8601.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:00:00 | 8595.00 | 8591.47 | 8601.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 8585.35 | 8590.25 | 8600.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:15:00 | 8568.25 | 8590.25 | 8600.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:00:00 | 8578.80 | 8585.84 | 8596.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 11:45:00 | 8572.15 | 8556.60 | 8577.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:30:00 | 8558.65 | 8535.85 | 8558.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 8436.85 | 8479.56 | 8515.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 11:30:00 | 8410.00 | 8457.92 | 8498.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:30:00 | 8415.60 | 8446.43 | 8486.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 8406.10 | 8447.46 | 8479.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:45:00 | 8417.90 | 8448.50 | 8471.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 8383.95 | 8417.40 | 8447.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 8320.00 | 8420.85 | 8438.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:00:00 | 8365.85 | 8409.85 | 8431.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 12:30:00 | 8353.10 | 8397.65 | 8421.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 14:45:00 | 8361.00 | 8395.05 | 8416.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 8407.85 | 8401.04 | 8415.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 8686.20 | 8470.15 | 8442.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 8686.20 | 8470.15 | 8442.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 8820.00 | 8712.65 | 8614.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 8709.15 | 8711.95 | 8623.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 8709.15 | 8711.95 | 8623.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 8979.80 | 9015.66 | 8926.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 8979.80 | 9015.66 | 8926.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 8967.75 | 9006.08 | 8930.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:30:00 | 8935.50 | 9006.08 | 8930.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 8960.00 | 8996.86 | 8933.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 8940.00 | 8996.86 | 8933.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 8927.85 | 8983.06 | 8932.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 8927.85 | 8983.06 | 8932.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 8900.00 | 8966.45 | 8929.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 8924.15 | 8966.45 | 8929.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 8957.05 | 8964.57 | 8932.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:30:00 | 9036.65 | 8972.72 | 8941.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 13:45:00 | 9007.05 | 8984.46 | 8952.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 15:15:00 | 8922.75 | 8938.25 | 8940.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 15:15:00 | 8922.75 | 8938.25 | 8940.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 8813.85 | 8913.37 | 8928.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 8905.65 | 8882.41 | 8902.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 8905.65 | 8882.41 | 8902.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 8905.65 | 8882.41 | 8902.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 8905.25 | 8882.41 | 8902.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 8938.60 | 8893.65 | 8906.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 8938.60 | 8893.65 | 8906.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 8976.00 | 8910.12 | 8912.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 9009.20 | 8910.12 | 8912.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 13:15:00 | 8927.00 | 8915.56 | 8914.62 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 8885.65 | 8920.79 | 8921.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 8852.55 | 8907.14 | 8915.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 8791.80 | 8761.51 | 8810.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 8791.80 | 8761.51 | 8810.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 8791.80 | 8761.51 | 8810.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 8791.80 | 8761.51 | 8810.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 8759.90 | 8742.48 | 8774.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 8759.90 | 8742.48 | 8774.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 8766.60 | 8747.31 | 8773.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:45:00 | 8770.35 | 8747.31 | 8773.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 8675.00 | 8732.85 | 8764.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 8665.50 | 8726.28 | 8758.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 8624.55 | 8517.18 | 8508.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 8624.55 | 8517.18 | 8508.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 8628.10 | 8574.49 | 8541.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 8496.90 | 8567.86 | 8545.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 8496.90 | 8567.86 | 8545.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 8496.90 | 8567.86 | 8545.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 8496.90 | 8567.86 | 8545.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 8489.80 | 8552.25 | 8540.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 8510.85 | 8546.20 | 8538.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 13:15:00 | 8494.10 | 8531.34 | 8532.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 8494.10 | 8531.34 | 8532.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 8379.65 | 8493.13 | 8514.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 15:15:00 | 8460.00 | 8450.15 | 8478.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 15:15:00 | 8460.00 | 8450.15 | 8478.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 8460.00 | 8450.15 | 8478.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 8420.65 | 8450.15 | 8478.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:30:00 | 8427.50 | 8442.25 | 8467.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 12:15:00 | 8006.12 | 8135.53 | 8250.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 13:15:00 | 7999.62 | 8107.21 | 8227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-04 09:15:00 | 7578.59 | 7740.18 | 7921.99 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 13:15:00 | 7581.70 | 7533.24 | 7528.33 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 7475.20 | 7519.40 | 7524.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 7416.40 | 7485.57 | 7506.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 7468.50 | 7453.32 | 7483.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:00:00 | 7468.50 | 7453.32 | 7483.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 7500.00 | 7456.37 | 7472.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:45:00 | 7500.00 | 7456.37 | 7472.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 7509.85 | 7467.07 | 7476.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 7464.15 | 7467.07 | 7476.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 7547.00 | 7490.37 | 7485.63 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 7460.75 | 7488.03 | 7488.38 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 15:15:00 | 7500.05 | 7489.39 | 7487.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 7528.15 | 7497.14 | 7491.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 10:15:00 | 7495.10 | 7496.73 | 7491.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 10:15:00 | 7495.10 | 7496.73 | 7491.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 7495.10 | 7496.73 | 7491.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 7492.25 | 7496.73 | 7491.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 7484.70 | 7494.33 | 7491.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 7491.80 | 7494.33 | 7491.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 7456.20 | 7486.70 | 7488.07 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 7569.90 | 7502.04 | 7494.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 7620.95 | 7557.01 | 7526.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 8013.95 | 8094.38 | 8021.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 8013.95 | 8094.38 | 8021.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 8013.95 | 8094.38 | 8021.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 8013.95 | 8094.38 | 8021.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 8015.00 | 8078.51 | 8020.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 8015.00 | 8078.51 | 8020.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 7980.30 | 8058.87 | 8017.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 7959.90 | 8058.87 | 8017.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 8069.00 | 8060.89 | 8021.93 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 7991.40 | 8007.68 | 8008.07 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 13:15:00 | 8036.10 | 8008.17 | 8007.91 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 7977.30 | 8002.00 | 8005.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 7969.00 | 7995.40 | 8001.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 11:15:00 | 8004.50 | 7989.30 | 7996.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 11:15:00 | 8004.50 | 7989.30 | 7996.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 8004.50 | 7989.30 | 7996.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 8002.35 | 7989.30 | 7996.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 7999.55 | 7991.35 | 7997.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 7999.55 | 7991.35 | 7997.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 7981.40 | 7989.36 | 7995.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 7998.85 | 7989.36 | 7995.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 7967.20 | 7984.93 | 7993.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 7967.20 | 7984.93 | 7993.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 7960.00 | 7979.94 | 7990.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 7938.95 | 7971.73 | 7985.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 7957.20 | 7971.30 | 7982.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 7952.00 | 7971.30 | 7982.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:00:00 | 7927.05 | 7931.01 | 7950.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 7978.05 | 7940.42 | 7952.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:00:00 | 7978.05 | 7940.42 | 7952.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 7970.50 | 7946.43 | 7954.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 7995.10 | 7964.66 | 7961.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 7995.10 | 7964.66 | 7961.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 8023.55 | 7998.00 | 7981.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 7882.55 | 7997.35 | 7987.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 7882.55 | 7997.35 | 7987.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 7882.55 | 7997.35 | 7987.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 7884.00 | 7997.35 | 7987.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 7881.80 | 7974.24 | 7978.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 7675.35 | 7877.41 | 7924.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 7443.75 | 7404.29 | 7558.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 10:15:00 | 7479.90 | 7404.29 | 7558.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 7495.25 | 7477.08 | 7526.89 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 7576.45 | 7547.82 | 7546.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 7759.35 | 7590.12 | 7566.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 7896.50 | 7924.05 | 7820.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:00:00 | 7896.50 | 7924.05 | 7820.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 7921.50 | 7921.28 | 7874.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 7961.00 | 7921.28 | 7874.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 7994.50 | 8145.68 | 8155.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 7994.50 | 8145.68 | 8155.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 7942.00 | 8007.43 | 8035.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 7890.00 | 7887.03 | 7951.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 13:15:00 | 7937.00 | 7901.16 | 7937.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 7937.00 | 7901.16 | 7937.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 7937.00 | 7901.16 | 7937.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 7922.50 | 7905.43 | 7936.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 7939.50 | 7905.43 | 7936.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 8019.00 | 7928.87 | 7941.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:30:00 | 8017.50 | 7928.87 | 7941.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 7993.50 | 7941.80 | 7946.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:30:00 | 8022.00 | 7941.80 | 7946.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 7994.00 | 7952.24 | 7950.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 12:15:00 | 8015.50 | 7964.89 | 7956.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 7955.00 | 7967.73 | 7959.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 7955.00 | 7967.73 | 7959.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 7955.00 | 7967.73 | 7959.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 7955.00 | 7967.73 | 7959.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 7926.50 | 7959.48 | 7956.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 7972.50 | 7959.48 | 7956.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 7872.50 | 7942.09 | 7948.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 13:15:00 | 7847.50 | 7913.02 | 7932.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 7711.50 | 7710.57 | 7764.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 7905.00 | 7710.57 | 7764.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 7919.50 | 7752.35 | 7778.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 7919.50 | 7752.35 | 7778.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 7932.00 | 7788.28 | 7792.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:30:00 | 7939.50 | 7788.28 | 7792.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 7926.00 | 7815.83 | 7804.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 8000.00 | 7852.66 | 7822.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 8068.00 | 8072.39 | 8011.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 8068.00 | 8072.39 | 8011.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 8606.50 | 8725.33 | 8607.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 8606.50 | 8725.33 | 8607.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 8557.00 | 8691.66 | 8602.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 8557.00 | 8691.66 | 8602.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 8600.00 | 8673.33 | 8602.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 8573.00 | 8673.33 | 8602.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 8735.00 | 8685.66 | 8614.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:15:00 | 8750.00 | 8685.66 | 8614.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 8531.00 | 8628.96 | 8622.45 | SL hit (close<static) qty=1.00 sl=8535.50 alert=retest2 |

### Cycle 153 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 8819.00 | 8858.95 | 8864.31 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 8899.00 | 8870.57 | 8868.88 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 8662.00 | 8828.86 | 8850.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 8608.00 | 8701.96 | 8771.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 10:15:00 | 8535.00 | 8529.76 | 8606.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 10:30:00 | 8539.00 | 8529.76 | 8606.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 8586.00 | 8553.03 | 8599.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 8565.00 | 8554.92 | 8595.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 8616.00 | 8569.39 | 8595.56 | SL hit (close>static) qty=1.00 sl=8600.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 8634.00 | 8580.76 | 8580.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 8743.00 | 8651.98 | 8631.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 8696.00 | 8708.88 | 8678.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 8696.00 | 8708.88 | 8678.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 8696.00 | 8708.88 | 8678.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 8696.00 | 8708.88 | 8678.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 8684.50 | 8704.01 | 8678.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 8675.00 | 8704.01 | 8678.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 8664.50 | 8696.11 | 8677.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 8664.50 | 8696.11 | 8677.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 8675.00 | 8691.88 | 8677.23 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 8561.50 | 8649.03 | 8659.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 8550.00 | 8629.22 | 8649.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 8500.00 | 8492.21 | 8547.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:15:00 | 8514.00 | 8492.21 | 8547.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 8566.00 | 8506.97 | 8549.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 8566.00 | 8506.97 | 8549.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 8554.50 | 8516.47 | 8549.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:00:00 | 8530.00 | 8534.96 | 8551.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 8496.50 | 8535.57 | 8550.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 8527.50 | 8513.33 | 8530.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 8635.00 | 8534.05 | 8534.89 | SL hit (close>static) qty=1.00 sl=8595.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 8556.50 | 8538.54 | 8536.86 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 8521.00 | 8533.50 | 8534.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 8447.50 | 8513.28 | 8525.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 10:15:00 | 8542.50 | 8519.08 | 8525.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 8542.50 | 8519.08 | 8525.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 8542.50 | 8519.08 | 8525.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 8542.50 | 8519.08 | 8525.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 8542.00 | 8523.66 | 8527.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 8517.00 | 8523.66 | 8527.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 8500.00 | 8518.93 | 8524.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 8494.00 | 8518.93 | 8524.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 8493.00 | 8512.62 | 8520.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 8397.50 | 8383.83 | 8383.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 8397.50 | 8383.83 | 8383.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 8428.00 | 8395.00 | 8388.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 8388.50 | 8395.86 | 8390.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 8388.50 | 8395.86 | 8390.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 8388.50 | 8395.86 | 8390.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 8388.50 | 8395.86 | 8390.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 8370.50 | 8390.79 | 8388.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 8361.50 | 8390.79 | 8388.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 8411.50 | 8394.93 | 8390.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 8422.50 | 8394.93 | 8390.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 14:15:00 | 8380.00 | 8419.81 | 8422.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 8380.00 | 8419.81 | 8422.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 8338.50 | 8390.76 | 8407.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 8404.50 | 8392.10 | 8405.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 8404.50 | 8392.10 | 8405.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 8404.50 | 8392.10 | 8405.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 8404.50 | 8392.10 | 8405.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 8389.50 | 8391.58 | 8403.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:45:00 | 8370.00 | 8387.78 | 8399.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 8362.00 | 8380.83 | 8394.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 8428.00 | 8375.68 | 8383.85 | SL hit (close>static) qty=1.00 sl=8409.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 8406.50 | 8388.94 | 8388.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 8437.00 | 8398.55 | 8393.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 8385.50 | 8399.61 | 8394.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 8385.50 | 8399.61 | 8394.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 8385.50 | 8399.61 | 8394.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:45:00 | 8385.00 | 8399.61 | 8394.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 8392.50 | 8398.19 | 8394.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 8393.00 | 8398.19 | 8394.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 8414.00 | 8412.38 | 8403.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 8414.00 | 8412.38 | 8403.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 8401.00 | 8410.11 | 8403.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:00:00 | 8401.00 | 8410.11 | 8403.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 8434.00 | 8414.89 | 8406.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 8441.50 | 8421.57 | 8410.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 8367.50 | 8433.75 | 8426.35 | SL hit (close<static) qty=1.00 sl=8400.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 8309.50 | 8408.90 | 8415.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 15:15:00 | 8271.50 | 8315.32 | 8342.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 8135.00 | 8103.64 | 8155.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 8135.00 | 8103.64 | 8155.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 8135.00 | 8103.64 | 8155.37 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 8247.00 | 8189.43 | 8184.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 8315.00 | 8214.55 | 8196.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 8210.00 | 8229.71 | 8207.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 8210.00 | 8229.71 | 8207.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 8210.00 | 8229.71 | 8207.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 8209.50 | 8229.71 | 8207.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 8266.50 | 8237.07 | 8212.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 8270.00 | 8244.85 | 8218.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 8301.00 | 8345.10 | 8349.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8301.00 | 8345.10 | 8349.47 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 8385.00 | 8351.32 | 8350.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 8395.50 | 8360.16 | 8354.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 8376.00 | 8379.76 | 8368.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 8376.00 | 8379.76 | 8368.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 8376.00 | 8379.76 | 8368.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 8382.50 | 8379.76 | 8368.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 8355.00 | 8374.80 | 8366.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 8355.50 | 8374.80 | 8366.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 8360.00 | 8371.84 | 8366.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 8360.00 | 8371.84 | 8366.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 8351.50 | 8367.77 | 8365.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 8351.50 | 8367.77 | 8365.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 8315.00 | 8357.22 | 8360.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 8285.50 | 8342.88 | 8353.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 8104.50 | 8101.65 | 8145.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 8089.00 | 8101.65 | 8145.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 8123.50 | 8113.93 | 8140.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 8123.50 | 8113.93 | 8140.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 8127.00 | 8116.54 | 8139.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 8106.50 | 8116.54 | 8139.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:45:00 | 8107.00 | 8116.23 | 8137.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:00:00 | 8115.50 | 8057.04 | 8059.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 8152.00 | 8076.03 | 8067.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 8152.00 | 8076.03 | 8067.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 8179.00 | 8108.46 | 8084.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 8186.00 | 8212.05 | 8179.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 13:15:00 | 8186.00 | 8212.05 | 8179.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 8186.00 | 8212.05 | 8179.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 8186.00 | 8212.05 | 8179.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 8182.00 | 8206.04 | 8179.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 8176.50 | 8206.04 | 8179.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 8175.00 | 8199.83 | 8179.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 8133.00 | 8199.83 | 8179.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 8126.50 | 8185.16 | 8174.71 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 8045.00 | 8157.13 | 8162.92 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 8221.00 | 8166.68 | 8164.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 8245.00 | 8182.35 | 8171.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 8203.50 | 8210.34 | 8192.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 13:15:00 | 8203.50 | 8210.34 | 8192.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 8203.50 | 8210.34 | 8192.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 8203.50 | 8210.34 | 8192.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 8225.50 | 8213.37 | 8195.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:15:00 | 8218.00 | 8213.37 | 8195.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 8218.00 | 8214.30 | 8197.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 8200.50 | 8207.14 | 8195.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 8146.00 | 8194.91 | 8190.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 8146.00 | 8194.91 | 8190.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 8169.00 | 8189.73 | 8188.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 8159.00 | 8189.73 | 8188.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 8252.50 | 8203.93 | 8195.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:30:00 | 8258.50 | 8217.94 | 8202.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 8282.00 | 8221.35 | 8205.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 8258.50 | 8246.67 | 8224.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 8263.00 | 8244.83 | 8225.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 8196.00 | 8235.07 | 8223.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 8196.00 | 8235.07 | 8223.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 8186.00 | 8225.25 | 8219.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 8186.00 | 8225.25 | 8219.78 | SL hit (close<static) qty=1.00 sl=8194.50 alert=retest2 |

### Cycle 171 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 8229.00 | 8233.13 | 8233.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 8210.50 | 8225.30 | 8229.39 | Break + close below crossover candle low |

### Cycle 172 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 8573.00 | 8294.84 | 8260.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 8783.00 | 8551.73 | 8426.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 8794.00 | 8806.33 | 8712.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 8745.00 | 8806.33 | 8712.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 8715.50 | 8783.47 | 8718.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 8715.50 | 8783.47 | 8718.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 8649.50 | 8756.68 | 8712.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 8649.50 | 8756.68 | 8712.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 8718.50 | 8749.04 | 8712.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 8639.00 | 8749.04 | 8712.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 8679.00 | 8735.03 | 8709.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 8679.00 | 8735.03 | 8709.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 8690.00 | 8726.03 | 8707.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 8651.00 | 8726.03 | 8707.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 8607.00 | 8680.30 | 8688.79 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 8735.00 | 8693.24 | 8690.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 8750.00 | 8711.91 | 8700.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 8668.50 | 8740.42 | 8727.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 8668.50 | 8740.42 | 8727.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 8668.50 | 8740.42 | 8727.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 8668.50 | 8740.42 | 8727.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 8676.00 | 8727.53 | 8722.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 8691.50 | 8727.53 | 8722.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 8701.00 | 8717.38 | 8719.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 8701.00 | 8717.38 | 8719.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 8625.50 | 8690.99 | 8706.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 8732.50 | 8673.45 | 8686.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 8732.50 | 8673.45 | 8686.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 8732.50 | 8673.45 | 8686.62 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 8847.00 | 8708.16 | 8701.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8749.23 | 8720.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 9056.50 | 9115.84 | 9062.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 9056.50 | 9115.84 | 9062.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 9056.50 | 9115.84 | 9062.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 9056.50 | 9115.84 | 9062.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 9095.00 | 9111.67 | 9065.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 9113.50 | 9111.67 | 9065.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 9126.50 | 9107.24 | 9067.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 9132.50 | 9090.53 | 9075.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 9232.50 | 9278.42 | 9283.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 9232.50 | 9278.42 | 9283.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 9171.50 | 9257.04 | 9272.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 9029.00 | 9017.28 | 9071.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:00:00 | 9029.00 | 9017.28 | 9071.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 9066.50 | 9029.24 | 9067.60 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 9111.00 | 9077.32 | 9074.53 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 9067.00 | 9075.55 | 9075.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 9033.50 | 9065.93 | 9071.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 9075.50 | 9066.18 | 9070.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 9075.50 | 9066.18 | 9070.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 9075.50 | 9066.18 | 9070.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 9075.50 | 9066.18 | 9070.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 9080.00 | 9068.94 | 9071.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 9004.00 | 9068.94 | 9071.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 9109.00 | 9026.99 | 9038.99 | SL hit (close>static) qty=1.00 sl=9080.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 9130.00 | 9047.59 | 9047.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 9175.50 | 9073.18 | 9058.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 9063.00 | 9093.01 | 9073.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 9063.00 | 9093.01 | 9073.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 9063.00 | 9093.01 | 9073.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 9074.00 | 9093.01 | 9073.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 9033.00 | 9081.01 | 9069.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 9143.00 | 9081.01 | 9069.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 9103.00 | 9085.41 | 9072.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 9072.00 | 9081.33 | 9072.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 8998.50 | 9055.96 | 9062.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 8998.50 | 9055.96 | 9062.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 8992.50 | 9043.27 | 9056.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 13:15:00 | 8867.00 | 8860.12 | 8913.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:45:00 | 8865.50 | 8860.12 | 8913.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 8835.50 | 8855.20 | 8906.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 8827.50 | 8855.20 | 8906.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 8815.00 | 8842.73 | 8891.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:00:00 | 8815.00 | 8838.30 | 8877.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 8806.00 | 8710.47 | 8742.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 8684.00 | 8705.18 | 8737.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 8664.50 | 8698.14 | 8731.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:30:00 | 8644.00 | 8687.21 | 8723.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:30:00 | 8638.00 | 8625.41 | 8660.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 8664.00 | 8644.80 | 8663.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 8722.00 | 8668.43 | 8671.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 8722.00 | 8668.43 | 8671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 8700.00 | 8674.75 | 8674.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 8700.00 | 8674.75 | 8674.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 8727.00 | 8685.20 | 8679.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 8804.00 | 8838.17 | 8786.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 8804.00 | 8838.17 | 8786.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 8816.50 | 8833.84 | 8789.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 8823.00 | 8833.84 | 8789.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 8797.50 | 8825.73 | 8799.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 8797.50 | 8825.73 | 8799.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 8769.00 | 8814.39 | 8796.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 8778.00 | 8814.39 | 8796.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 8723.50 | 8796.21 | 8790.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 8723.50 | 8796.21 | 8790.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 8782.00 | 8793.37 | 8789.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 8795.00 | 8793.49 | 8789.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 8792.00 | 8796.40 | 8792.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 9120.50 | 9125.25 | 9125.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 9120.50 | 9125.25 | 9125.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 9078.00 | 9105.94 | 9115.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 13:15:00 | 9073.50 | 9067.62 | 9085.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:00:00 | 9073.50 | 9067.62 | 9085.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 9062.00 | 9066.50 | 9083.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 9040.00 | 9068.25 | 9078.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 9039.00 | 9065.88 | 9074.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 9003.50 | 9064.11 | 9072.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:00:00 | 9035.50 | 9055.70 | 9065.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 9000.50 | 9035.68 | 9052.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 8957.50 | 9017.39 | 9038.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 8893.00 | 8966.22 | 8997.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 15:15:00 | 8770.00 | 8750.02 | 8747.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 8770.00 | 8750.02 | 8747.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 8866.00 | 8779.61 | 8762.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 8815.50 | 8857.49 | 8828.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 8815.50 | 8857.49 | 8828.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 8815.50 | 8857.49 | 8828.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 8815.50 | 8857.49 | 8828.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 8870.50 | 8860.09 | 8832.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 8876.50 | 8857.07 | 8833.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:45:00 | 8886.50 | 8863.26 | 8838.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 8882.00 | 8868.01 | 8843.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:30:00 | 8873.50 | 8867.71 | 8845.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 8861.50 | 8864.94 | 8850.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 8804.00 | 8864.94 | 8850.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 8815.00 | 8854.95 | 8847.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 8802.00 | 8844.36 | 8843.48 | SL hit (close<static) qty=1.00 sl=8814.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 8810.50 | 8837.59 | 8840.48 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 8980.00 | 8864.81 | 8851.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 9003.00 | 8911.52 | 8875.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 8917.00 | 8958.45 | 8932.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 8917.00 | 8958.45 | 8932.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 8917.00 | 8958.45 | 8932.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 8917.00 | 8958.45 | 8932.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 8915.00 | 8949.76 | 8930.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 8957.50 | 8949.76 | 8930.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 8920.00 | 8942.73 | 8930.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 8920.00 | 8942.73 | 8930.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 8926.00 | 8939.38 | 8930.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:15:00 | 8926.50 | 8939.38 | 8930.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 8929.50 | 8937.40 | 8930.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:15:00 | 8891.00 | 8937.40 | 8930.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 8900.00 | 8929.92 | 8927.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:45:00 | 8876.50 | 8929.92 | 8927.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 8884.00 | 8920.74 | 8923.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 8868.50 | 8910.29 | 8918.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 8918.00 | 8911.83 | 8918.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 8918.00 | 8911.83 | 8918.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 8918.00 | 8911.83 | 8918.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 8916.00 | 8911.83 | 8918.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 8997.00 | 8928.87 | 8925.55 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 8900.50 | 8932.49 | 8934.36 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 9006.00 | 8942.86 | 8935.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 09:15:00 | 9052.00 | 9001.66 | 8972.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 14:15:00 | 9045.00 | 9057.45 | 9015.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 9045.00 | 9057.45 | 9015.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 9050.00 | 9124.57 | 9091.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 9050.00 | 9124.57 | 9091.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 9022.00 | 9104.06 | 9084.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 9022.00 | 9104.06 | 9084.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 9040.00 | 9067.26 | 9070.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 8955.50 | 9044.91 | 9060.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 9034.00 | 9024.08 | 9044.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 9034.00 | 9024.08 | 9044.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 9042.00 | 9027.66 | 9044.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:15:00 | 9060.00 | 9027.66 | 9044.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 9073.50 | 9036.83 | 9047.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 9073.50 | 9036.83 | 9047.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 9080.00 | 9045.47 | 9050.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 9089.50 | 9045.47 | 9050.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 9162.50 | 9068.87 | 9060.44 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 9023.00 | 9065.78 | 9068.49 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 9088.50 | 9068.09 | 9067.64 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 9014.00 | 9057.90 | 9063.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 8980.00 | 9029.52 | 9047.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 9059.00 | 9022.25 | 9036.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 9059.00 | 9022.25 | 9036.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 9059.00 | 9022.25 | 9036.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 9070.00 | 9022.25 | 9036.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 9035.00 | 9024.80 | 9036.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:30:00 | 9017.00 | 9027.84 | 9036.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 9082.50 | 9049.25 | 9045.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 9082.50 | 9049.25 | 9045.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 9097.00 | 9069.67 | 9056.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 9016.00 | 9078.66 | 9069.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 9016.00 | 9078.66 | 9069.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 9016.00 | 9078.66 | 9069.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 9016.00 | 9078.66 | 9069.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 9041.50 | 9071.23 | 9067.17 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 9004.00 | 9057.79 | 9061.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 8980.00 | 9038.58 | 9051.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 13:15:00 | 8968.50 | 8962.16 | 8986.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 8968.50 | 8962.16 | 8986.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 8993.00 | 8968.33 | 8987.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 8993.00 | 8968.33 | 8987.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 8980.00 | 8970.67 | 8986.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 8988.50 | 8970.67 | 8986.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 9031.00 | 8982.73 | 8990.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 9031.00 | 8982.73 | 8990.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 9038.50 | 8993.89 | 8995.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 9038.50 | 8993.89 | 8995.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 9080.00 | 9011.11 | 9002.90 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 8925.00 | 8997.16 | 9006.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 10:15:00 | 8900.00 | 8977.73 | 8997.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 8958.50 | 8948.80 | 8967.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 8958.50 | 8948.80 | 8967.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 8958.50 | 8948.80 | 8967.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 8960.50 | 8948.80 | 8967.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 8974.50 | 8953.94 | 8968.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 8974.50 | 8953.94 | 8968.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 8990.00 | 8961.15 | 8970.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:45:00 | 8990.50 | 8961.15 | 8970.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 9014.50 | 8971.82 | 8974.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 9014.50 | 8971.82 | 8974.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 8989.00 | 8975.26 | 8975.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 8961.50 | 8970.61 | 8973.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 8963.50 | 8900.26 | 8898.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 8963.50 | 8900.26 | 8898.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 8999.00 | 8932.13 | 8913.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 9094.00 | 9110.80 | 9056.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 9094.00 | 9110.80 | 9056.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 9110.50 | 9106.11 | 9071.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 9156.00 | 9110.39 | 9076.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 9150.00 | 9147.68 | 9114.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 9065.00 | 9112.52 | 9108.77 | SL hit (close<static) qty=1.00 sl=9070.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 9069.00 | 9103.82 | 9105.15 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 9166.50 | 9104.79 | 9102.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 9207.00 | 9125.23 | 9112.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 9421.00 | 9448.15 | 9372.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:45:00 | 9416.50 | 9448.15 | 9372.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 9484.50 | 9536.46 | 9482.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 9484.50 | 9536.46 | 9482.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 9499.00 | 9521.45 | 9484.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 9650.00 | 9521.45 | 9484.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 9563.00 | 9684.36 | 9700.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 9563.00 | 9684.36 | 9700.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 9555.00 | 9658.49 | 9686.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 9558.50 | 9509.00 | 9546.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 9558.50 | 9509.00 | 9546.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 9558.50 | 9509.00 | 9546.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 9558.50 | 9509.00 | 9546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 9550.00 | 9517.20 | 9546.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 9603.00 | 9517.20 | 9546.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 9541.00 | 9521.96 | 9545.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 9566.00 | 9521.96 | 9545.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 9570.50 | 9531.67 | 9548.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 9570.50 | 9531.67 | 9548.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 9539.00 | 9533.14 | 9547.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 9561.50 | 9533.14 | 9547.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 9549.50 | 9536.41 | 9547.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 9517.00 | 9535.03 | 9545.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 9518.50 | 9535.03 | 9545.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 14:15:00 | 9588.50 | 9545.72 | 9549.79 | SL hit (close>static) qty=1.00 sl=9562.50 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 9559.50 | 9552.36 | 9552.35 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 9471.00 | 9536.09 | 9544.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 9403.50 | 9493.64 | 9522.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 9291.00 | 9211.69 | 9268.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 9291.00 | 9211.69 | 9268.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 9291.00 | 9211.69 | 9268.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 9304.50 | 9211.69 | 9268.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 9212.50 | 9211.85 | 9263.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 9188.00 | 9212.88 | 9259.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 9368.00 | 9262.99 | 9272.31 | SL hit (close>static) qty=1.00 sl=9297.50 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 9387.00 | 9287.80 | 9282.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 9411.50 | 9312.54 | 9294.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 9390.00 | 9392.34 | 9352.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 9398.50 | 9392.34 | 9352.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 9400.00 | 9393.87 | 9356.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:00:00 | 9503.00 | 9424.11 | 9386.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 9270.50 | 9405.53 | 9385.17 | SL hit (close<static) qty=1.00 sl=9330.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 9280.50 | 9461.21 | 9475.97 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 9654.50 | 9494.35 | 9479.35 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 9517.50 | 9587.97 | 9595.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 9494.00 | 9569.18 | 9586.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 9592.50 | 9552.64 | 9570.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 9592.50 | 9552.64 | 9570.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 9592.50 | 9552.64 | 9570.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 9592.50 | 9552.64 | 9570.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 9574.50 | 9557.01 | 9570.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 9572.00 | 9557.81 | 9570.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 9710.00 | 9597.10 | 9584.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 9710.00 | 9597.10 | 9584.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 9761.00 | 9629.88 | 9600.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 13:15:00 | 9836.00 | 9866.06 | 9809.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 9836.00 | 9866.06 | 9809.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 9773.00 | 9838.47 | 9809.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 9765.00 | 9838.47 | 9809.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 9790.50 | 9828.88 | 9808.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 9854.00 | 9832.30 | 9811.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:30:00 | 9808.50 | 9823.01 | 9810.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 9748.00 | 9798.41 | 9801.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 9748.00 | 9798.41 | 9801.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 9675.00 | 9762.86 | 9783.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 9769.00 | 9725.84 | 9750.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 9769.00 | 9725.84 | 9750.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 9769.00 | 9725.84 | 9750.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 9769.00 | 9725.84 | 9750.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 9753.00 | 9731.27 | 9750.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 9765.00 | 9731.27 | 9750.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 9825.00 | 9750.02 | 9757.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 9825.00 | 9750.02 | 9757.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 9830.00 | 9766.02 | 9764.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 9889.50 | 9812.36 | 9788.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 9906.00 | 9914.08 | 9865.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 9920.50 | 9914.08 | 9865.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 9852.50 | 9901.77 | 9863.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 9852.50 | 9901.77 | 9863.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 9837.00 | 9888.81 | 9861.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 9813.50 | 9888.81 | 9861.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 9726.50 | 9845.74 | 9846.03 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 9871.50 | 9821.37 | 9818.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 9905.00 | 9858.72 | 9839.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 9785.50 | 9848.20 | 9837.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 9785.50 | 9848.20 | 9837.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 9785.50 | 9848.20 | 9837.92 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 9780.00 | 9824.53 | 9828.32 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 9921.00 | 9844.03 | 9835.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 10121.50 | 9899.53 | 9861.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 10033.50 | 10073.35 | 10020.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 10038.00 | 10073.35 | 10020.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 9985.50 | 10055.78 | 10017.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 9967.50 | 10055.78 | 10017.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 9994.50 | 10043.53 | 10015.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:00:00 | 10042.00 | 10043.22 | 10017.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 10035.50 | 10041.68 | 10019.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 9973.00 | 10027.94 | 10015.30 | SL hit (close<static) qty=1.00 sl=9976.50 alert=retest2 |

### Cycle 217 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 9770.00 | 9964.76 | 9988.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 9672.50 | 9840.31 | 9919.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 9704.00 | 9676.03 | 9751.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 9704.00 | 9676.03 | 9751.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 9704.00 | 9676.03 | 9751.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 9648.50 | 9676.28 | 9738.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 9784.00 | 9707.18 | 9738.26 | SL hit (close>static) qty=1.00 sl=9773.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 9849.00 | 9766.80 | 9759.78 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 9482.00 | 9736.92 | 9753.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 9388.50 | 9667.24 | 9720.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 9577.00 | 9491.73 | 9579.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 9577.00 | 9491.73 | 9579.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 9505.00 | 9494.38 | 9572.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 9499.50 | 9494.38 | 9572.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 9602.50 | 9531.70 | 9571.67 | SL hit (close>static) qty=1.00 sl=9599.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 9147.50 | 9084.73 | 9079.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 9212.50 | 9117.02 | 9096.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 9079.50 | 9185.46 | 9153.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 9079.50 | 9185.46 | 9153.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 9079.50 | 9185.46 | 9153.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 9059.00 | 9185.46 | 9153.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 9001.00 | 9112.61 | 9125.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 8883.50 | 9066.79 | 9103.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 9006.50 | 9000.23 | 9058.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 9006.50 | 9000.23 | 9058.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 9006.50 | 9000.23 | 9058.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 8910.00 | 9041.62 | 9057.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 8870.00 | 8854.20 | 8928.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 8919.00 | 8897.45 | 8922.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 9081.50 | 8934.99 | 8935.57 | SL hit (close>static) qty=1.00 sl=9074.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 9138.50 | 8975.69 | 8954.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 9139.50 | 9008.45 | 8970.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 8894.50 | 9013.29 | 8992.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 8894.50 | 9013.29 | 8992.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 8894.50 | 9013.29 | 8992.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 8900.50 | 9013.29 | 8992.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 8865.00 | 8983.63 | 8981.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 8854.50 | 8983.63 | 8981.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 8877.00 | 8962.30 | 8971.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 8837.50 | 8916.00 | 8944.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 9008.50 | 8865.93 | 8894.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 9008.50 | 8865.93 | 8894.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 9008.50 | 8865.93 | 8894.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 9046.50 | 8865.93 | 8894.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 8972.50 | 8912.03 | 8911.03 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 15:15:00 | 8893.00 | 8911.61 | 8911.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 8698.00 | 8868.89 | 8892.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 8815.00 | 8784.10 | 8836.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 8815.00 | 8784.10 | 8836.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 8769.50 | 8781.18 | 8830.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 8778.00 | 8781.18 | 8830.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 8954.50 | 8809.25 | 8834.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 8954.50 | 8809.25 | 8834.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 8867.00 | 8820.80 | 8837.31 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 8942.00 | 8856.83 | 8851.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 8975.00 | 8928.53 | 8897.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 10:15:00 | 9720.00 | 9727.82 | 9578.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 11:00:00 | 9720.00 | 9727.82 | 9578.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 9752.00 | 9819.70 | 9776.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 9752.00 | 9819.70 | 9776.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 9776.50 | 9811.06 | 9776.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 9820.00 | 9813.95 | 9780.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 9750.00 | 9775.74 | 9775.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 9750.00 | 9775.74 | 9775.95 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 9799.00 | 9777.39 | 9776.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 9833.00 | 9790.93 | 9782.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 9740.00 | 9792.48 | 9786.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 9740.00 | 9792.48 | 9786.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 9740.00 | 9792.48 | 9786.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 9858.50 | 9805.68 | 9793.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:45:00 | 9824.50 | 9806.55 | 9794.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 9752.50 | 9790.04 | 9791.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 9752.50 | 9790.04 | 9791.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 9699.50 | 9761.22 | 9777.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 9634.00 | 9585.78 | 9642.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 9634.00 | 9585.78 | 9642.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 9634.00 | 9585.78 | 9642.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:00:00 | 9634.00 | 9585.78 | 9642.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 9592.50 | 9587.12 | 9637.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 9617.50 | 9587.12 | 9637.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 9618.00 | 9593.30 | 9635.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 9638.50 | 9593.30 | 9635.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 9558.00 | 9586.24 | 9628.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 9539.50 | 9577.79 | 9621.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 9668.00 | 9595.86 | 9618.51 | SL hit (close>static) qty=1.00 sl=9637.50 alert=retest2 |

### Cycle 230 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 9704.00 | 9635.74 | 9632.50 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 9543.50 | 9628.62 | 9634.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 9536.00 | 9596.95 | 9617.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 9644.50 | 9574.54 | 9599.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 9644.50 | 9574.54 | 9599.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 9644.50 | 9574.54 | 9599.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 9674.00 | 9574.54 | 9599.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 9626.00 | 9584.83 | 9602.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 9596.00 | 9596.31 | 9604.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 9678.00 | 9560.53 | 9580.24 | SL hit (close>static) qty=1.00 sl=9653.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 9755.00 | 9599.42 | 9596.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 9835.00 | 9646.54 | 9617.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 10067.00 | 10087.60 | 9962.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 11:00:00 | 10067.00 | 10087.60 | 9962.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-19 14:00:00 | 4477.85 | 2023-05-22 10:15:00 | 4551.05 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-05-30 09:15:00 | 4628.10 | 2023-05-30 12:15:00 | 4602.75 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-06-15 12:30:00 | 4727.55 | 2023-06-21 09:15:00 | 4689.15 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2023-06-15 14:00:00 | 4723.70 | 2023-06-21 09:15:00 | 4689.15 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2023-06-23 11:15:00 | 4619.95 | 2023-06-27 09:15:00 | 4670.05 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-06-23 13:00:00 | 4620.00 | 2023-06-27 09:15:00 | 4670.05 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-06-23 15:00:00 | 4621.50 | 2023-06-27 09:15:00 | 4670.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-06-26 11:00:00 | 4620.00 | 2023-06-27 09:15:00 | 4670.05 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-07-10 09:15:00 | 4909.10 | 2023-07-13 13:15:00 | 4861.45 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-07-19 14:15:00 | 4825.00 | 2023-07-20 15:15:00 | 4880.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-07-20 10:45:00 | 4822.05 | 2023-07-20 15:15:00 | 4880.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-08-03 13:15:00 | 4822.85 | 2023-08-11 15:15:00 | 4581.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-04 09:15:00 | 4817.50 | 2023-08-14 09:15:00 | 4576.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-03 13:15:00 | 4822.85 | 2023-08-14 13:15:00 | 4606.35 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2023-08-04 09:15:00 | 4817.50 | 2023-08-14 13:15:00 | 4606.35 | STOP_HIT | 0.50 | 4.38% |
| BUY | retest2 | 2023-08-25 09:15:00 | 4639.50 | 2023-08-25 10:15:00 | 4606.25 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2023-08-31 09:15:00 | 4682.45 | 2023-08-31 12:15:00 | 4655.45 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-09-14 09:30:00 | 4827.30 | 2023-09-21 15:15:00 | 5084.20 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2023-09-14 12:45:00 | 4820.90 | 2023-09-21 15:15:00 | 5084.20 | STOP_HIT | 1.00 | 5.46% |
| BUY | retest2 | 2023-09-14 13:45:00 | 4821.25 | 2023-09-21 15:15:00 | 5084.20 | STOP_HIT | 1.00 | 5.45% |
| SELL | retest2 | 2023-09-28 11:15:00 | 5035.00 | 2023-09-29 13:15:00 | 5080.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-10-19 09:15:00 | 5348.25 | 2023-10-25 14:15:00 | 5349.95 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2023-11-09 10:30:00 | 5450.80 | 2023-11-10 09:15:00 | 5360.15 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2023-11-21 09:15:00 | 5698.35 | 2023-11-28 14:15:00 | 5983.27 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-21 09:15:00 | 5698.35 | 2023-11-30 09:15:00 | 6032.00 | STOP_HIT | 0.50 | 5.86% |
| BUY | retest2 | 2023-12-18 09:15:00 | 6403.50 | 2023-12-21 09:15:00 | 6284.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-01-11 09:15:00 | 7102.70 | 2024-01-17 09:15:00 | 7105.65 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-01-20 15:15:00 | 7084.00 | 2024-01-23 09:15:00 | 7191.45 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-01-23 11:45:00 | 7108.40 | 2024-01-24 13:15:00 | 7181.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-01-24 12:00:00 | 7119.20 | 2024-01-24 13:15:00 | 7181.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-01-31 12:15:00 | 7633.20 | 2024-02-07 11:15:00 | 7687.70 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-02-09 12:15:00 | 7753.90 | 2024-02-19 09:15:00 | 8529.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-09 13:45:00 | 7759.00 | 2024-02-19 09:15:00 | 8534.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-15 10:30:00 | 8223.40 | 2024-03-18 11:15:00 | 8515.00 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-03-21 09:45:00 | 8664.90 | 2024-04-03 15:15:00 | 9084.45 | STOP_HIT | 1.00 | 4.84% |
| SELL | retest2 | 2024-04-09 13:15:00 | 8989.50 | 2024-04-12 10:15:00 | 9099.25 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-04-10 09:45:00 | 8945.95 | 2024-04-12 10:15:00 | 9099.25 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-04-10 14:45:00 | 8987.45 | 2024-04-12 10:15:00 | 9099.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-04-24 13:00:00 | 8775.05 | 2024-04-26 09:15:00 | 8890.60 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-04-24 14:30:00 | 8759.90 | 2024-04-26 09:15:00 | 8890.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-06 11:15:00 | 8980.85 | 2024-05-07 09:15:00 | 8935.80 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-05-09 12:30:00 | 8822.05 | 2024-05-10 09:15:00 | 8973.60 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-05-09 14:15:00 | 8833.80 | 2024-05-10 09:15:00 | 8973.60 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-05-09 15:00:00 | 8837.50 | 2024-05-10 09:15:00 | 8973.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-05-17 10:45:00 | 8810.00 | 2024-05-23 12:15:00 | 8888.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-05-18 12:15:00 | 8808.75 | 2024-05-23 12:15:00 | 8888.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-05-21 14:30:00 | 8816.55 | 2024-05-23 12:15:00 | 8888.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-05-21 15:15:00 | 8800.00 | 2024-05-23 12:15:00 | 8888.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-05-22 12:30:00 | 8752.70 | 2024-05-23 12:15:00 | 8888.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-05-27 11:45:00 | 8984.50 | 2024-06-04 11:15:00 | 8911.90 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-05-28 11:00:00 | 9008.00 | 2024-06-04 11:15:00 | 8911.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-05-28 11:30:00 | 8988.50 | 2024-06-04 11:15:00 | 8911.90 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-28 13:00:00 | 8980.00 | 2024-06-04 13:15:00 | 9064.00 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2024-05-29 12:45:00 | 8983.00 | 2024-06-04 13:15:00 | 9064.00 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-05-30 15:15:00 | 8999.15 | 2024-06-04 13:15:00 | 9064.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-05-31 12:45:00 | 8987.95 | 2024-06-04 13:15:00 | 9064.00 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2024-06-04 12:30:00 | 9024.05 | 2024-06-04 13:15:00 | 9064.00 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-06-14 10:15:00 | 9898.50 | 2024-06-19 10:15:00 | 9779.85 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-06-19 09:30:00 | 9897.85 | 2024-06-19 10:15:00 | 9779.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest1 | 2024-06-28 10:00:00 | 9403.30 | 2024-06-28 13:15:00 | 9500.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-07-03 12:30:00 | 9407.35 | 2024-07-05 09:15:00 | 9530.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-07-10 09:15:00 | 9620.00 | 2024-07-10 10:15:00 | 9478.45 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-07-12 12:30:00 | 9443.55 | 2024-07-15 09:15:00 | 9590.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-07-12 13:30:00 | 9453.35 | 2024-07-15 09:15:00 | 9590.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-08-02 10:45:00 | 9717.40 | 2024-08-05 09:15:00 | 9445.15 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-08-09 09:15:00 | 9752.95 | 2024-08-28 11:15:00 | 10632.27 | TARGET_HIT | 1.00 | 9.02% |
| BUY | retest2 | 2024-08-09 12:30:00 | 9665.70 | 2024-08-28 11:15:00 | 10641.35 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2024-08-12 10:30:00 | 9673.95 | 2024-08-29 09:15:00 | 10728.25 | TARGET_HIT | 1.00 | 10.90% |
| BUY | retest1 | 2024-09-17 10:15:00 | 11779.30 | 2024-09-18 13:15:00 | 11786.20 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2024-09-20 09:15:00 | 11926.00 | 2024-09-30 12:15:00 | 12367.75 | STOP_HIT | 1.00 | 3.70% |
| SELL | retest2 | 2024-10-08 14:30:00 | 11825.55 | 2024-10-08 15:15:00 | 11913.50 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-10-11 09:15:00 | 11922.75 | 2024-10-11 09:15:00 | 11766.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-10-28 14:30:00 | 10074.00 | 2024-11-04 09:15:00 | 9570.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-28 14:30:00 | 10074.00 | 2024-11-05 09:15:00 | 9645.00 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2024-11-07 12:15:00 | 9898.30 | 2024-11-11 13:15:00 | 9858.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-11-07 13:45:00 | 9896.00 | 2024-11-11 13:15:00 | 9858.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-11-08 09:15:00 | 9915.20 | 2024-11-11 13:15:00 | 9858.95 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-11-08 10:30:00 | 9893.00 | 2024-11-12 09:15:00 | 9851.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-11-08 13:15:00 | 9925.00 | 2024-11-12 10:15:00 | 9820.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-11-08 14:30:00 | 9914.20 | 2024-11-12 10:15:00 | 9820.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-11-11 09:30:00 | 9963.60 | 2024-11-12 10:15:00 | 9820.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-11-11 14:30:00 | 9920.55 | 2024-11-12 10:15:00 | 9820.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-11-25 15:00:00 | 9400.05 | 2024-12-03 10:15:00 | 9206.00 | STOP_HIT | 1.00 | 2.06% |
| SELL | retest2 | 2024-12-17 09:15:00 | 8970.10 | 2024-12-19 13:15:00 | 8981.20 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-12-17 11:00:00 | 8973.95 | 2024-12-19 15:15:00 | 8980.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-12-18 10:45:00 | 8959.95 | 2024-12-19 15:15:00 | 8980.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-12-18 11:30:00 | 8974.15 | 2024-12-19 15:15:00 | 8980.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-12-19 09:15:00 | 8836.30 | 2024-12-19 15:15:00 | 8980.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-12-30 11:30:00 | 8885.00 | 2024-12-30 12:15:00 | 8818.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-01-10 09:45:00 | 8664.95 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-01-13 09:15:00 | 8672.55 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-01-13 10:45:00 | 8696.60 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-01-15 09:45:00 | 8660.00 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-01-17 13:15:00 | 8568.25 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-01-17 15:00:00 | 8578.80 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-01-20 11:45:00 | 8572.15 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-01-21 09:30:00 | 8558.65 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-01-22 11:30:00 | 8410.00 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-01-22 13:30:00 | 8415.60 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-01-23 09:15:00 | 8406.10 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-01-23 12:45:00 | 8417.90 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-01-27 10:15:00 | 8320.00 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-01-27 11:00:00 | 8365.85 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-01-27 12:30:00 | 8353.10 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-01-27 14:45:00 | 8361.00 | 2025-01-29 09:15:00 | 8686.20 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-02-04 11:30:00 | 9036.65 | 2025-02-05 15:15:00 | 8922.75 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-02-04 13:45:00 | 9007.05 | 2025-02-05 15:15:00 | 8922.75 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-02-13 15:15:00 | 8665.50 | 2025-02-20 10:15:00 | 8624.55 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-02-21 11:30:00 | 8510.85 | 2025-02-21 13:15:00 | 8494.10 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-02-25 09:15:00 | 8420.65 | 2025-02-28 12:15:00 | 8006.12 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-02-25 11:30:00 | 8427.50 | 2025-02-28 13:15:00 | 7999.62 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-02-25 09:15:00 | 8420.65 | 2025-03-04 09:15:00 | 7578.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 11:30:00 | 8427.50 | 2025-03-04 09:15:00 | 7584.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 09:30:00 | 7938.95 | 2025-04-01 14:15:00 | 7995.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-03-28 11:30:00 | 7957.20 | 2025-04-01 14:15:00 | 7995.10 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-03-28 12:15:00 | 7952.00 | 2025-04-01 14:15:00 | 7995.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-04-01 11:00:00 | 7927.05 | 2025-04-01 14:15:00 | 7995.10 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-04-17 11:15:00 | 7961.00 | 2025-04-25 10:15:00 | 7994.50 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-05-21 10:15:00 | 8750.00 | 2025-05-22 10:15:00 | 8531.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-05-22 15:15:00 | 8744.00 | 2025-05-29 13:15:00 | 8819.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-05-23 14:15:00 | 8764.00 | 2025-05-29 13:15:00 | 8819.00 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-05-23 15:00:00 | 8744.00 | 2025-05-29 13:15:00 | 8819.00 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-06-03 14:45:00 | 8565.00 | 2025-06-04 09:15:00 | 8616.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-04 11:45:00 | 8561.50 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-04 14:45:00 | 8565.00 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-05 10:30:00 | 8564.50 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-05 14:00:00 | 8549.00 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-06 09:30:00 | 8552.00 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-16 15:00:00 | 8530.00 | 2025-06-18 09:15:00 | 8635.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-17 09:15:00 | 8496.50 | 2025-06-18 09:15:00 | 8635.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-17 14:15:00 | 8527.50 | 2025-06-18 09:15:00 | 8635.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-06-19 13:15:00 | 8494.00 | 2025-06-25 12:15:00 | 8397.50 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-06-19 15:15:00 | 8493.00 | 2025-06-25 12:15:00 | 8397.50 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2025-06-26 14:15:00 | 8422.50 | 2025-06-30 14:15:00 | 8380.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-07-02 09:45:00 | 8370.00 | 2025-07-03 09:15:00 | 8428.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-02 10:30:00 | 8362.00 | 2025-07-03 09:15:00 | 8428.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-07 09:30:00 | 8441.50 | 2025-07-08 09:15:00 | 8367.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-16 11:30:00 | 8270.00 | 2025-07-22 14:15:00 | 8301.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-07-30 09:15:00 | 8106.50 | 2025-08-04 10:15:00 | 8152.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-30 09:45:00 | 8107.00 | 2025-08-04 10:15:00 | 8152.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-04 10:00:00 | 8115.50 | 2025-08-04 10:15:00 | 8152.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-11 14:30:00 | 8258.50 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-12 09:15:00 | 8282.00 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-12 12:30:00 | 8258.50 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-12 13:45:00 | 8263.00 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-08-13 09:15:00 | 8217.00 | 2025-08-14 13:15:00 | 8229.00 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-08-13 09:45:00 | 8260.00 | 2025-08-14 13:15:00 | 8229.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-08-28 09:15:00 | 8691.50 | 2025-08-28 13:15:00 | 8701.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-09-05 09:15:00 | 9113.50 | 2025-09-10 15:15:00 | 9232.50 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-09-05 10:15:00 | 9126.50 | 2025-09-10 15:15:00 | 9232.50 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-09-08 09:15:00 | 9132.50 | 2025-09-10 15:15:00 | 9232.50 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-09-19 09:15:00 | 9004.00 | 2025-09-22 09:15:00 | 9109.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-23 09:15:00 | 9143.00 | 2025-09-23 13:15:00 | 8998.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-23 10:00:00 | 9103.00 | 2025-09-23 13:15:00 | 8998.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-23 11:15:00 | 9072.00 | 2025-09-23 13:15:00 | 8998.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-25 15:15:00 | 8827.50 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2025-09-26 10:00:00 | 8815.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-09-26 13:00:00 | 8815.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-10-01 09:30:00 | 8806.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-10-01 11:45:00 | 8664.50 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-01 12:30:00 | 8644.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-03 13:30:00 | 8638.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-06 09:15:00 | 8664.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-09 11:45:00 | 8795.00 | 2025-10-23 10:15:00 | 9120.50 | STOP_HIT | 1.00 | 3.70% |
| BUY | retest2 | 2025-10-09 14:30:00 | 8792.00 | 2025-10-23 10:15:00 | 9120.50 | STOP_HIT | 1.00 | 3.74% |
| SELL | retest2 | 2025-10-28 12:00:00 | 9040.00 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.99% |
| SELL | retest2 | 2025-10-28 14:45:00 | 9039.00 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.98% |
| SELL | retest2 | 2025-10-29 09:15:00 | 9003.50 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.59% |
| SELL | retest2 | 2025-10-29 13:00:00 | 9035.50 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.94% |
| SELL | retest2 | 2025-10-30 13:15:00 | 8957.50 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-10-31 12:45:00 | 8893.00 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-11-13 09:15:00 | 8876.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-13 09:45:00 | 8886.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-13 10:45:00 | 8882.00 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-13 11:30:00 | 8873.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-04 11:30:00 | 9017.00 | 2025-12-04 14:15:00 | 9082.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-17 09:30:00 | 8961.50 | 2025-12-19 12:15:00 | 8963.50 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-12-24 11:15:00 | 9156.00 | 2025-12-26 14:15:00 | 9065.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-26 09:45:00 | 9150.00 | 2025-12-26 14:15:00 | 9065.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-06 09:15:00 | 9650.00 | 2026-01-09 13:15:00 | 9563.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-14 13:45:00 | 9517.00 | 2026-01-14 14:15:00 | 9588.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-14 14:15:00 | 9518.50 | 2026-01-14 14:15:00 | 9588.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-22 11:45:00 | 9188.00 | 2026-01-22 14:15:00 | 9368.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-01-27 15:00:00 | 9503.00 | 2026-01-28 09:15:00 | 9270.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-29 09:15:00 | 9513.50 | 2026-02-02 09:15:00 | 9280.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-29 15:00:00 | 9513.00 | 2026-02-02 09:15:00 | 9280.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-30 11:45:00 | 9488.00 | 2026-02-02 09:15:00 | 9280.50 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-02-09 11:45:00 | 9572.00 | 2026-02-10 09:15:00 | 9710.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-02-13 11:45:00 | 9854.00 | 2026-02-13 15:15:00 | 9748.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-13 13:30:00 | 9808.50 | 2026-02-13 15:15:00 | 9748.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-27 13:00:00 | 10042.00 | 2026-02-27 14:15:00 | 9973.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-02-27 14:00:00 | 10035.50 | 2026-02-27 14:15:00 | 9973.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-27 14:30:00 | 10035.50 | 2026-02-27 15:15:00 | 9955.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-03-05 11:45:00 | 9648.50 | 2026-03-05 14:15:00 | 9784.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-03-10 12:15:00 | 9499.50 | 2026-03-10 14:15:00 | 9602.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-03-11 10:00:00 | 9498.00 | 2026-03-13 09:15:00 | 9023.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:00:00 | 9498.00 | 2026-03-16 11:15:00 | 8925.00 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-03-23 09:15:00 | 8910.00 | 2026-03-25 09:15:00 | 9081.50 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-03-24 10:00:00 | 8870.00 | 2026-03-25 09:15:00 | 9081.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-03-24 14:30:00 | 8919.00 | 2026-03-25 09:15:00 | 9081.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-16 14:30:00 | 9820.00 | 2026-04-17 15:15:00 | 9750.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-04-21 10:00:00 | 9858.50 | 2026-04-22 09:15:00 | 9752.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-04-21 10:45:00 | 9824.50 | 2026-04-22 09:15:00 | 9752.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-04-24 13:30:00 | 9539.50 | 2026-04-27 09:15:00 | 9668.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-29 13:15:00 | 9596.00 | 2026-04-30 10:15:00 | 9678.00 | STOP_HIT | 1.00 | -0.85% |
