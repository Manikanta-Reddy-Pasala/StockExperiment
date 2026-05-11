# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 4700.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 157 |
| ALERT1 | 109 |
| ALERT2 | 107 |
| ALERT2_SKIP | 58 |
| ALERT3 | 263 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 129 |
| PARTIAL | 23 |
| TARGET_HIT | 8 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 159 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 102
- **Target hits / Stop hits / Partials:** 8 / 128 / 23
- **Avg / median % per leg:** 0.50% / -0.97%
- **Sum % (uncompounded):** 80.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 8 | 11.3% | 3 | 68 | 0 | -1.22% | -86.6% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.07% | -6.4% |
| BUY @ 3rd Alert (retest2) | 65 | 8 | 12.3% | 3 | 62 | 0 | -1.23% | -80.1% |
| SELL (all) | 88 | 49 | 55.7% | 5 | 60 | 23 | 1.89% | 166.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.61% | -1.6% |
| SELL @ 3rd Alert (retest2) | 87 | 49 | 56.3% | 5 | 59 | 23 | 1.94% | 168.4% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.15% | -8.0% |
| retest2 (combined) | 152 | 57 | 37.5% | 8 | 121 | 23 | 0.58% | 88.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 4650.00 | 4557.91 | 4546.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 4689.45 | 4584.21 | 4559.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 14:15:00 | 4650.40 | 4689.54 | 4644.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 14:15:00 | 4650.40 | 4689.54 | 4644.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 4650.40 | 4689.54 | 4644.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:30:00 | 4759.80 | 4700.31 | 4657.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 11:45:00 | 4749.00 | 4719.49 | 4674.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 13:15:00 | 4748.45 | 4720.64 | 4678.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 15:00:00 | 4743.60 | 4729.80 | 4690.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 4771.10 | 4739.68 | 4701.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-16 13:15:00 | 4595.05 | 4681.30 | 4684.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 4595.05 | 4681.30 | 4684.21 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 4762.20 | 4684.62 | 4683.70 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 4674.90 | 4712.31 | 4712.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 4625.05 | 4694.86 | 4704.90 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 4887.45 | 4716.10 | 4709.02 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 4649.05 | 4702.69 | 4703.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 4567.20 | 4675.59 | 4691.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 4694.90 | 4602.29 | 4640.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 4694.90 | 4602.29 | 4640.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 4694.90 | 4602.29 | 4640.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 4705.65 | 4602.29 | 4640.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 4570.70 | 4595.97 | 4634.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 12:00:00 | 4543.10 | 4585.40 | 4625.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:00:00 | 4546.25 | 4566.13 | 4595.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 4552.30 | 4563.73 | 4592.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:30:00 | 4532.35 | 4563.96 | 4587.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 4428.20 | 4536.81 | 4573.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 4309.70 | 4479.49 | 4523.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 4315.94 | 4436.74 | 4499.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 4318.94 | 4436.74 | 4499.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 4324.69 | 4436.74 | 4499.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 4305.73 | 4436.74 | 4499.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 4248.00 | 4225.79 | 4291.69 | SL hit (close>ema200) qty=0.50 sl=4225.79 alert=retest2 |

### Cycle 7 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 4500.00 | 4250.12 | 4248.16 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 4507.15 | 4541.26 | 4542.25 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 11:15:00 | 4548.70 | 4540.67 | 4539.66 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 12:15:00 | 4521.50 | 4536.84 | 4538.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 4502.80 | 4530.03 | 4534.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 14:15:00 | 4545.70 | 4533.16 | 4535.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 14:15:00 | 4545.70 | 4533.16 | 4535.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 4545.70 | 4533.16 | 4535.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 4545.70 | 4533.16 | 4535.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 4550.05 | 4536.54 | 4537.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 4586.95 | 4536.54 | 4537.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 4612.30 | 4551.69 | 4543.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 13:15:00 | 4674.45 | 4612.10 | 4587.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 4680.10 | 4686.77 | 4654.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 4680.10 | 4686.77 | 4654.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 4680.10 | 4686.77 | 4654.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 4676.70 | 4686.77 | 4654.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 4713.40 | 4730.01 | 4696.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 4711.00 | 4730.01 | 4696.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 4822.40 | 4858.27 | 4830.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 4890.00 | 4858.27 | 4830.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 4781.75 | 4837.63 | 4829.90 | SL hit (close<static) qty=1.00 sl=4810.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 4751.10 | 4820.32 | 4822.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 14:15:00 | 4723.55 | 4800.97 | 4813.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 14:15:00 | 4705.05 | 4702.23 | 4748.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 4705.05 | 4702.23 | 4748.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 4595.05 | 4546.35 | 4584.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 4595.05 | 4546.35 | 4584.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 4590.00 | 4555.08 | 4584.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 4650.05 | 4555.08 | 4584.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 4615.30 | 4567.12 | 4587.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 11:30:00 | 4593.05 | 4575.47 | 4588.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:00:00 | 4583.25 | 4575.47 | 4588.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 4591.10 | 4579.66 | 4588.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 4591.50 | 4583.73 | 4589.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 4598.00 | 4586.58 | 4590.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:45:00 | 4598.50 | 4586.58 | 4590.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 4600.80 | 4591.25 | 4592.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 4601.80 | 4593.36 | 4593.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 4601.80 | 4593.36 | 4593.10 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 4586.90 | 4592.65 | 4592.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 4584.00 | 4590.92 | 4592.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 12:15:00 | 4586.25 | 4583.43 | 4587.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 12:15:00 | 4586.25 | 4583.43 | 4587.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 4586.25 | 4583.43 | 4587.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:00:00 | 4586.25 | 4583.43 | 4587.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 4555.75 | 4577.89 | 4584.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 4555.75 | 4577.89 | 4584.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 4645.85 | 4591.48 | 4590.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 15:15:00 | 4675.00 | 4608.19 | 4598.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 12:15:00 | 4620.55 | 4626.68 | 4612.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 13:00:00 | 4620.55 | 4626.68 | 4612.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 4572.05 | 4615.76 | 4608.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:45:00 | 4569.90 | 4615.76 | 4608.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 4564.55 | 4605.51 | 4604.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 4564.55 | 4605.51 | 4604.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 4510.00 | 4586.41 | 4595.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 12:15:00 | 4497.85 | 4545.02 | 4571.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 4567.00 | 4537.46 | 4562.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 4567.00 | 4537.46 | 4562.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 4567.00 | 4537.46 | 4562.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 4567.00 | 4537.46 | 4562.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 4570.00 | 4543.97 | 4563.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 4484.35 | 4543.97 | 4563.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:15:00 | 4260.13 | 4358.84 | 4424.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 4286.95 | 4256.31 | 4305.20 | SL hit (close>ema200) qty=0.50 sl=4256.31 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 4447.65 | 4273.71 | 4255.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 14:15:00 | 4749.85 | 4419.43 | 4337.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 4468.00 | 4494.74 | 4437.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 10:00:00 | 4468.00 | 4494.74 | 4437.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 4462.95 | 4482.60 | 4452.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 4462.95 | 4482.60 | 4452.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 4450.80 | 4476.24 | 4452.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 4453.60 | 4476.24 | 4452.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 4433.60 | 4467.71 | 4450.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 4433.60 | 4467.71 | 4450.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 4430.30 | 4460.23 | 4448.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 4425.05 | 4460.23 | 4448.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 13:15:00 | 4420.00 | 4442.47 | 4442.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 09:15:00 | 4389.30 | 4424.98 | 4433.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 14:15:00 | 4352.70 | 4315.71 | 4347.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 14:15:00 | 4352.70 | 4315.71 | 4347.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 4352.70 | 4315.71 | 4347.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 4352.70 | 4315.71 | 4347.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 4349.00 | 4322.37 | 4347.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 4306.00 | 4322.37 | 4347.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 10:30:00 | 4324.85 | 4330.17 | 4337.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 4333.00 | 4324.39 | 4327.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 12:45:00 | 4305.50 | 4322.00 | 4326.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 4090.70 | 4247.33 | 4287.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 4108.61 | 4247.33 | 4287.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 4116.35 | 4247.33 | 4287.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 4090.22 | 4247.33 | 4287.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 4040.05 | 4131.34 | 4208.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 09:30:00 | 3992.95 | 4079.87 | 4170.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-06 13:15:00 | 3892.37 | 3994.18 | 4096.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 3870.10 | 3854.86 | 3854.18 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 3786.95 | 3845.66 | 3851.26 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 3930.00 | 3861.77 | 3853.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 10:15:00 | 3970.65 | 3883.55 | 3864.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 4180.35 | 4201.34 | 4129.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:00:00 | 4180.35 | 4201.34 | 4129.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 4217.90 | 4195.20 | 4148.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 4143.60 | 4195.20 | 4148.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 4282.70 | 4333.17 | 4294.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 4282.70 | 4333.17 | 4294.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 4280.00 | 4322.53 | 4293.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 4328.05 | 4322.53 | 4293.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-29 15:15:00 | 4760.86 | 4681.93 | 4602.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 4805.80 | 4913.52 | 4921.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 11:15:00 | 4786.30 | 4825.20 | 4847.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 12:15:00 | 4760.40 | 4757.42 | 4793.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 12:15:00 | 4760.40 | 4757.42 | 4793.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 4760.40 | 4757.42 | 4793.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:00:00 | 4760.40 | 4757.42 | 4793.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 4748.25 | 4718.49 | 4759.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:15:00 | 4788.25 | 4718.49 | 4759.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 4882.00 | 4751.19 | 4771.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 4882.00 | 4751.19 | 4771.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 4833.40 | 4767.64 | 4776.69 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 4880.15 | 4790.14 | 4786.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 4915.00 | 4829.84 | 4805.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 4843.05 | 4923.44 | 4877.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 4843.05 | 4923.44 | 4877.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 4843.05 | 4923.44 | 4877.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 4843.05 | 4923.44 | 4877.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 4830.00 | 4904.75 | 4873.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 4927.00 | 4904.75 | 4873.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:45:00 | 4866.95 | 4899.77 | 4883.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 13:15:00 | 4940.20 | 4970.69 | 4972.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 13:15:00 | 4940.20 | 4970.69 | 4972.61 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 14:15:00 | 5028.40 | 4982.23 | 4977.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 5035.35 | 5006.26 | 4992.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 5002.00 | 5005.41 | 4993.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:15:00 | 4995.50 | 5005.41 | 4993.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 4977.00 | 4999.73 | 4992.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:45:00 | 4993.50 | 4999.73 | 4992.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 5004.05 | 5000.59 | 4993.35 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 4942.20 | 4985.31 | 4987.43 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 5065.90 | 4992.64 | 4987.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 15:15:00 | 5137.00 | 5049.69 | 5018.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 5239.00 | 5274.19 | 5208.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 5243.05 | 5274.19 | 5208.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 5252.75 | 5266.39 | 5215.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 5245.35 | 5266.39 | 5215.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 5224.05 | 5257.93 | 5216.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 5224.05 | 5257.93 | 5216.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 5247.00 | 5255.74 | 5219.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 5168.70 | 5255.74 | 5219.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 5196.70 | 5242.47 | 5224.78 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 5200.05 | 5212.61 | 5213.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 14:15:00 | 5128.05 | 5195.70 | 5206.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 5302.45 | 5207.31 | 5208.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 5302.45 | 5207.31 | 5208.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 5302.45 | 5207.31 | 5208.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 5302.45 | 5207.31 | 5208.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 5279.90 | 5221.83 | 5215.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 14:15:00 | 5330.00 | 5265.55 | 5240.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 5231.10 | 5274.02 | 5251.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 10:15:00 | 5231.10 | 5274.02 | 5251.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 5231.10 | 5274.02 | 5251.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 5231.10 | 5274.02 | 5251.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 5319.65 | 5283.15 | 5257.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 5413.65 | 5305.98 | 5276.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 14:00:00 | 5369.55 | 5356.31 | 5316.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:45:00 | 5374.25 | 5366.21 | 5331.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 10:30:00 | 5371.05 | 5365.67 | 5334.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 5337.75 | 5360.09 | 5334.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:00:00 | 5337.75 | 5360.09 | 5334.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 5320.00 | 5352.07 | 5333.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:15:00 | 5304.10 | 5352.07 | 5333.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 5298.00 | 5341.25 | 5329.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:30:00 | 5295.05 | 5341.25 | 5329.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 5204.40 | 5303.71 | 5314.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 5204.40 | 5303.71 | 5314.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 5178.00 | 5278.57 | 5302.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 5100.60 | 5090.59 | 5154.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:45:00 | 5105.00 | 5090.59 | 5154.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 5149.90 | 5102.45 | 5153.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 5149.90 | 5102.45 | 5153.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 5130.00 | 5107.96 | 5151.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 14:00:00 | 5086.50 | 5107.20 | 5143.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 15:15:00 | 5160.00 | 5118.11 | 5142.54 | SL hit (close>static) qty=1.00 sl=5154.80 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 5115.00 | 4948.33 | 4938.38 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 4941.50 | 4979.19 | 4979.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 15:15:00 | 4927.40 | 4968.83 | 4974.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 4979.80 | 4971.02 | 4975.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 4979.80 | 4971.02 | 4975.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 4979.80 | 4971.02 | 4975.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 4979.80 | 4971.02 | 4975.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 4985.90 | 4974.00 | 4976.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 4942.70 | 4966.36 | 4971.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 13:15:00 | 4695.56 | 4817.47 | 4883.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 4925.50 | 4780.07 | 4815.85 | SL hit (close>ema200) qty=0.50 sl=4780.07 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 4860.20 | 4749.87 | 4743.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 4968.70 | 4847.39 | 4800.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 14:15:00 | 4829.95 | 4877.64 | 4837.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 4829.95 | 4877.64 | 4837.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 4829.95 | 4877.64 | 4837.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 4829.95 | 4877.64 | 4837.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 4775.20 | 4857.15 | 4831.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 4841.25 | 4853.97 | 4832.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 4972.90 | 4945.82 | 4901.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 4771.95 | 4881.76 | 4894.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 13:15:00 | 4771.95 | 4881.76 | 4894.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 4705.75 | 4797.06 | 4838.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 4268.00 | 4221.86 | 4303.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 4268.00 | 4221.86 | 4303.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 4268.00 | 4221.86 | 4303.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:30:00 | 4314.65 | 4221.86 | 4303.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 4199.45 | 4132.53 | 4162.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:45:00 | 4208.40 | 4132.53 | 4162.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 4180.35 | 4142.10 | 4164.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 4182.80 | 4142.10 | 4164.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 4195.30 | 4152.74 | 4166.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:30:00 | 4191.00 | 4152.74 | 4166.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 4165.00 | 4159.25 | 4167.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 14:00:00 | 4165.00 | 4159.25 | 4167.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 4105.00 | 4148.40 | 4161.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 14:45:00 | 4183.55 | 4148.40 | 4161.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 4195.35 | 4147.80 | 4158.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 4195.35 | 4147.80 | 4158.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 4208.05 | 4159.85 | 4163.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 4205.50 | 4159.85 | 4163.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 4223.95 | 4172.67 | 4168.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 4415.00 | 4233.17 | 4199.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 15:15:00 | 4274.65 | 4279.82 | 4243.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 15:15:00 | 4274.65 | 4279.82 | 4243.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 4274.65 | 4279.82 | 4243.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 4376.50 | 4279.82 | 4243.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:00:00 | 4319.30 | 4287.72 | 4250.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 4198.30 | 4254.84 | 4245.56 | SL hit (close<static) qty=1.00 sl=4236.50 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 4167.00 | 4237.27 | 4238.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 4118.35 | 4161.07 | 4186.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 4017.15 | 4013.59 | 4071.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 09:30:00 | 4019.65 | 4013.59 | 4071.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 4044.90 | 4026.49 | 4063.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 13:30:00 | 4016.85 | 4024.58 | 4059.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 4028.00 | 4025.67 | 4056.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 4170.00 | 4056.15 | 4058.45 | SL hit (close>static) qty=1.00 sl=4064.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 4121.40 | 4069.20 | 4064.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 15:15:00 | 4195.00 | 4154.01 | 4129.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 4370.00 | 4385.86 | 4337.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 11:30:00 | 4358.60 | 4385.86 | 4337.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 4339.50 | 4376.58 | 4337.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:00:00 | 4339.50 | 4376.58 | 4337.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 4355.10 | 4372.29 | 4339.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:00:00 | 4355.10 | 4372.29 | 4339.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 4366.95 | 4371.22 | 4341.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:30:00 | 4342.40 | 4371.22 | 4341.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 4328.25 | 4362.63 | 4340.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 4303.65 | 4362.63 | 4340.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 4266.35 | 4343.37 | 4333.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 4266.35 | 4343.37 | 4333.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 4270.20 | 4328.74 | 4327.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 4275.00 | 4328.74 | 4327.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 4285.40 | 4320.07 | 4324.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 4285.40 | 4320.07 | 4324.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 4218.00 | 4281.63 | 4302.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 4406.65 | 4258.69 | 4270.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 4406.65 | 4258.69 | 4270.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 4406.65 | 4258.69 | 4270.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 4406.65 | 4258.69 | 4270.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 4399.00 | 4286.75 | 4282.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 4442.55 | 4317.91 | 4296.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 4427.00 | 4427.73 | 4388.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:15:00 | 4372.70 | 4427.73 | 4388.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 4455.60 | 4433.31 | 4394.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 10:15:00 | 4475.30 | 4433.31 | 4394.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 11:00:00 | 4479.85 | 4442.61 | 4402.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 15:00:00 | 4477.35 | 4457.12 | 4423.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 4543.00 | 4455.46 | 4436.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 4434.30 | 4468.85 | 4449.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 4432.85 | 4468.85 | 4449.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 4402.90 | 4455.66 | 4445.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 4402.90 | 4455.66 | 4445.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 4344.80 | 4422.81 | 4431.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 4344.80 | 4422.81 | 4431.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 4270.30 | 4392.31 | 4416.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 4372.75 | 4362.13 | 4394.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 4372.75 | 4362.13 | 4394.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 4410.85 | 4368.68 | 4391.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:00:00 | 4410.85 | 4368.68 | 4391.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 4401.15 | 4375.18 | 4392.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:30:00 | 4423.00 | 4375.18 | 4392.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 4527.70 | 4405.68 | 4404.65 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 4379.55 | 4406.29 | 4409.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 4334.95 | 4387.61 | 4400.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 4361.05 | 4360.85 | 4382.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 4361.05 | 4360.85 | 4382.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 4395.00 | 4367.68 | 4383.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 4403.70 | 4367.68 | 4383.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 4388.85 | 4371.91 | 4384.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 4326.00 | 4380.92 | 4386.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:30:00 | 4339.95 | 4369.80 | 4378.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 4109.70 | 4303.49 | 4346.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 4122.95 | 4303.49 | 4346.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 4194.00 | 4181.25 | 4244.70 | SL hit (close>ema200) qty=0.50 sl=4181.25 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 4424.45 | 4241.08 | 4224.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 4501.05 | 4321.93 | 4266.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 4585.90 | 4619.65 | 4509.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:45:00 | 4584.90 | 4619.65 | 4509.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 4520.55 | 4583.80 | 4511.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 4502.40 | 4583.80 | 4511.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 4480.50 | 4563.14 | 4508.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:00:00 | 4480.50 | 4563.14 | 4508.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 4440.00 | 4538.51 | 4502.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 4434.60 | 4538.51 | 4502.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 4453.00 | 4510.22 | 4495.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:45:00 | 4493.80 | 4506.14 | 4495.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 15:15:00 | 4426.00 | 4492.51 | 4494.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 15:15:00 | 4426.00 | 4492.51 | 4494.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 4417.35 | 4466.40 | 4481.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 4644.70 | 4450.11 | 4458.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 4644.70 | 4450.11 | 4458.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 4644.70 | 4450.11 | 4458.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 4644.70 | 4450.11 | 4458.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 4725.05 | 4505.10 | 4482.84 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 4436.75 | 4542.83 | 4547.96 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 12:15:00 | 4610.00 | 4540.69 | 4532.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 4629.05 | 4574.64 | 4551.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 14:15:00 | 4609.50 | 4630.36 | 4595.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 15:00:00 | 4609.50 | 4630.36 | 4595.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 4590.00 | 4622.29 | 4595.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 4656.95 | 4622.29 | 4595.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 4556.85 | 4605.30 | 4591.91 | SL hit (close<static) qty=1.00 sl=4560.05 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 14:15:00 | 4472.05 | 4568.50 | 4577.40 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 4631.80 | 4571.02 | 4570.49 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 4499.80 | 4611.30 | 4617.31 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 4607.10 | 4587.87 | 4587.21 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 4529.70 | 4579.05 | 4585.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 15:15:00 | 4505.05 | 4560.42 | 4575.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 4434.55 | 4425.62 | 4457.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 4434.55 | 4425.62 | 4457.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 4434.55 | 4425.62 | 4457.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 4420.40 | 4425.37 | 4454.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:45:00 | 4420.00 | 4424.29 | 4451.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:45:00 | 4409.40 | 4411.46 | 4433.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 4575.05 | 4437.28 | 4433.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 4575.05 | 4437.28 | 4433.20 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 4397.45 | 4499.38 | 4500.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 4358.50 | 4439.89 | 4470.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 4618.10 | 4432.72 | 4452.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 4618.10 | 4432.72 | 4452.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 4618.10 | 4432.72 | 4452.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 4618.10 | 4432.72 | 4452.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 4613.00 | 4468.78 | 4466.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 4681.70 | 4593.35 | 4540.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 4599.00 | 4636.12 | 4586.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 4599.00 | 4636.12 | 4586.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 4604.00 | 4629.70 | 4588.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:30:00 | 4644.55 | 4638.73 | 4596.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 4640.55 | 4658.16 | 4634.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 4653.20 | 4648.56 | 4632.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:00:00 | 4665.00 | 4645.41 | 4634.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 4646.90 | 4645.70 | 4635.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 4605.95 | 4645.70 | 4635.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 4543.55 | 4625.27 | 4627.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 4543.55 | 4625.27 | 4627.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 4520.00 | 4594.54 | 4612.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 4403.45 | 4394.45 | 4459.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:30:00 | 4395.20 | 4394.45 | 4459.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 4471.30 | 4412.50 | 4456.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 4471.30 | 4412.50 | 4456.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 4422.45 | 4414.49 | 4453.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 4467.25 | 4414.49 | 4453.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 4470.75 | 4425.74 | 4455.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 4470.75 | 4425.74 | 4455.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 4439.00 | 4428.39 | 4453.72 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 13:15:00 | 4492.80 | 4470.17 | 4467.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 14:15:00 | 4496.20 | 4475.38 | 4470.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 4428.85 | 4470.01 | 4469.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 4428.85 | 4470.01 | 4469.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 4428.85 | 4470.01 | 4469.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:45:00 | 4442.15 | 4470.01 | 4469.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 4406.60 | 4457.33 | 4463.37 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 14:15:00 | 4570.00 | 4468.44 | 4464.70 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 10:15:00 | 4406.35 | 4458.09 | 4461.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 12:15:00 | 4383.80 | 4435.14 | 4450.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 4561.15 | 4455.80 | 4456.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 14:15:00 | 4561.15 | 4455.80 | 4456.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 4561.15 | 4455.80 | 4456.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 4561.15 | 4455.80 | 4456.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 4572.00 | 4479.04 | 4467.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 4658.10 | 4545.69 | 4503.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 4679.65 | 4706.72 | 4649.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 4656.55 | 4694.79 | 4669.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 4656.55 | 4694.79 | 4669.79 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 4571.65 | 4648.82 | 4657.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 4488.00 | 4569.21 | 4613.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 4483.95 | 4468.32 | 4522.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 15:00:00 | 4483.95 | 4468.32 | 4522.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 4525.40 | 4478.08 | 4517.23 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 15:15:00 | 4580.00 | 4542.01 | 4537.04 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 4474.50 | 4528.51 | 4531.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 10:15:00 | 4429.15 | 4508.64 | 4522.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 4549.55 | 4497.36 | 4510.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 4549.55 | 4497.36 | 4510.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 4549.55 | 4497.36 | 4510.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 4549.55 | 4497.36 | 4510.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 4544.60 | 4506.81 | 4513.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 4466.15 | 4506.81 | 4513.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 4450.55 | 4495.56 | 4507.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:30:00 | 4420.00 | 4476.30 | 4497.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 11:15:00 | 4550.55 | 4504.03 | 4498.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 4550.55 | 4504.03 | 4498.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 4602.50 | 4533.37 | 4513.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 09:15:00 | 4529.00 | 4548.38 | 4526.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 4529.00 | 4548.38 | 4526.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 4529.00 | 4548.38 | 4526.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 10:15:00 | 4588.20 | 4548.38 | 4526.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 13:15:00 | 4583.50 | 4560.05 | 4538.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 12:15:00 | 4502.50 | 4530.08 | 4533.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 12:15:00 | 4502.50 | 4530.08 | 4533.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 14:15:00 | 4449.50 | 4509.42 | 4522.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 4465.65 | 4452.63 | 4476.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 4465.65 | 4452.63 | 4476.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 4465.65 | 4452.63 | 4476.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 4500.00 | 4452.63 | 4476.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 4488.25 | 4459.76 | 4477.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 4488.25 | 4459.76 | 4477.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 4488.35 | 4465.47 | 4478.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:30:00 | 4491.70 | 4465.47 | 4478.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 4408.90 | 4459.25 | 4472.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 4398.85 | 4447.40 | 4466.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 15:15:00 | 4508.00 | 4455.93 | 4458.87 | SL hit (close>static) qty=1.00 sl=4485.75 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 4557.00 | 4476.14 | 4467.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 10:15:00 | 4615.00 | 4503.92 | 4481.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 4643.00 | 4646.82 | 4594.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 15:00:00 | 4643.00 | 4646.82 | 4594.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 4583.00 | 4634.06 | 4593.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 4677.00 | 4634.06 | 4593.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 15:15:00 | 4777.00 | 4867.46 | 4873.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 15:15:00 | 4777.00 | 4867.46 | 4873.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 4682.55 | 4785.46 | 4804.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 4542.15 | 4519.70 | 4573.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 4542.15 | 4519.70 | 4573.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 4501.95 | 4482.27 | 4510.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 4501.95 | 4482.27 | 4510.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 4486.10 | 4483.03 | 4508.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 4498.95 | 4483.03 | 4508.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 4482.00 | 4482.83 | 4505.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:30:00 | 4468.05 | 4488.77 | 4501.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 14:15:00 | 4571.70 | 4505.36 | 4507.48 | SL hit (close>static) qty=1.00 sl=4551.70 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 4561.50 | 4516.58 | 4512.39 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 4400.55 | 4493.38 | 4502.22 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 4571.95 | 4506.60 | 4498.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 4654.00 | 4536.08 | 4512.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 11:15:00 | 4691.70 | 4712.39 | 4663.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 12:00:00 | 4691.70 | 4712.39 | 4663.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 4647.40 | 4693.69 | 4666.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 4647.40 | 4693.69 | 4666.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 4635.00 | 4681.95 | 4663.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 4841.70 | 4681.95 | 4663.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 09:15:00 | 5325.87 | 5230.98 | 5174.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 13:15:00 | 5227.50 | 5256.25 | 5258.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 10:15:00 | 5216.10 | 5245.58 | 5252.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 5224.20 | 5210.63 | 5230.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 5224.20 | 5210.63 | 5230.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 5295.20 | 5227.54 | 5236.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 5273.10 | 5227.54 | 5236.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 5287.80 | 5239.59 | 5241.38 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 5290.00 | 5249.67 | 5245.80 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 5191.50 | 5238.04 | 5240.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 5092.00 | 5190.62 | 5213.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 09:15:00 | 5149.50 | 5133.96 | 5166.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 5149.50 | 5133.96 | 5166.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 5149.50 | 5133.96 | 5166.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 5151.50 | 5133.96 | 5166.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 5128.00 | 5132.77 | 5162.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 5114.50 | 5127.81 | 5157.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:45:00 | 5109.50 | 5124.15 | 5153.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 5223.50 | 5139.15 | 5149.86 | SL hit (close>static) qty=1.00 sl=5172.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 5261.50 | 5163.62 | 5160.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 5295.00 | 5215.85 | 5187.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 5186.50 | 5236.53 | 5205.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 5186.50 | 5236.53 | 5205.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 5186.50 | 5236.53 | 5205.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:45:00 | 5181.00 | 5236.53 | 5205.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 5148.50 | 5218.93 | 5200.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:45:00 | 5152.00 | 5218.93 | 5200.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 5151.00 | 5184.69 | 5187.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 5013.50 | 5150.45 | 5171.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 5021.50 | 5000.51 | 5063.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 5021.50 | 5000.51 | 5063.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 5021.50 | 5000.51 | 5063.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:45:00 | 5074.50 | 5000.51 | 5063.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 5071.00 | 5013.73 | 5058.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 12:00:00 | 5018.00 | 5023.43 | 5055.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 5045.00 | 5042.34 | 5054.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:45:00 | 5029.50 | 5037.47 | 5050.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 5141.00 | 5054.63 | 5048.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 5141.00 | 5054.63 | 5048.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 5170.00 | 5092.16 | 5067.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 5140.50 | 5145.85 | 5108.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 5140.50 | 5145.85 | 5108.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 5076.50 | 5129.04 | 5112.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 5076.50 | 5129.04 | 5112.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 5078.00 | 5118.83 | 5109.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 5078.00 | 5118.83 | 5109.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 10:15:00 | 5044.00 | 5092.21 | 5098.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 12:15:00 | 5035.00 | 5073.70 | 5088.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 5100.00 | 5072.76 | 5084.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 5100.00 | 5072.76 | 5084.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 5100.00 | 5072.76 | 5084.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 5100.00 | 5072.76 | 5084.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 5094.50 | 5077.11 | 5085.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 5095.00 | 5077.11 | 5085.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 5075.50 | 5076.79 | 5084.78 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 5124.00 | 5096.28 | 5092.62 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 5065.00 | 5087.10 | 5089.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 5059.00 | 5081.48 | 5086.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 13:15:00 | 5104.00 | 5078.23 | 5084.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 13:15:00 | 5104.00 | 5078.23 | 5084.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 5104.00 | 5078.23 | 5084.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 5104.00 | 5078.23 | 5084.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 5108.50 | 5084.28 | 5086.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 5100.50 | 5084.28 | 5086.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 5118.50 | 5091.08 | 5089.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 12:15:00 | 5182.50 | 5125.69 | 5106.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 5150.00 | 5151.01 | 5128.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 10:45:00 | 5144.50 | 5151.01 | 5128.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 5103.00 | 5141.41 | 5126.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:45:00 | 5093.50 | 5141.41 | 5126.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 5114.00 | 5135.93 | 5125.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:30:00 | 5107.00 | 5135.93 | 5125.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 5115.00 | 5128.43 | 5124.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 5121.00 | 5128.43 | 5124.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5172.00 | 5137.15 | 5128.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 5198.00 | 5143.92 | 5132.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 5201.50 | 5222.47 | 5187.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:30:00 | 5344.50 | 5229.09 | 5199.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 5114.00 | 5191.10 | 5200.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 5114.00 | 5191.10 | 5200.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 5074.00 | 5136.54 | 5168.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 5097.00 | 5066.00 | 5109.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 5097.00 | 5066.00 | 5109.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 5095.00 | 5071.80 | 5107.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 5103.00 | 5071.80 | 5107.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5077.50 | 5078.26 | 5102.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 5062.00 | 5071.01 | 5096.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 5072.00 | 5075.53 | 5094.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 5147.00 | 5092.94 | 5099.49 | SL hit (close>static) qty=1.00 sl=5125.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 5140.00 | 5109.65 | 5106.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 5149.50 | 5117.62 | 5110.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 5080.50 | 5133.19 | 5123.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 5080.50 | 5133.19 | 5123.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 5080.50 | 5133.19 | 5123.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 5080.50 | 5133.19 | 5123.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 5080.00 | 5122.55 | 5119.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:15:00 | 5075.50 | 5122.55 | 5119.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 5075.00 | 5113.04 | 5115.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 5045.50 | 5099.53 | 5108.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 5097.50 | 5057.65 | 5080.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 5097.50 | 5057.65 | 5080.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 5097.50 | 5057.65 | 5080.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 5097.50 | 5057.65 | 5080.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 5100.00 | 5066.12 | 5081.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:15:00 | 5078.00 | 5078.15 | 5084.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 5058.50 | 5074.22 | 5082.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 5075.50 | 5045.95 | 5052.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 5075.00 | 5056.41 | 5056.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 5075.00 | 5056.41 | 5056.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 5112.50 | 5069.88 | 5062.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 5089.50 | 5092.28 | 5080.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 5089.50 | 5092.28 | 5080.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 5089.50 | 5092.28 | 5080.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 5087.00 | 5092.28 | 5080.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 5096.00 | 5093.02 | 5081.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 5080.00 | 5093.02 | 5081.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 5087.00 | 5091.82 | 5082.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 5088.50 | 5091.82 | 5082.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 5065.00 | 5086.45 | 5080.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 5065.00 | 5086.45 | 5080.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 5068.00 | 5082.76 | 5079.65 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 5046.00 | 5072.81 | 5075.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 5031.50 | 5060.68 | 5068.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 12:15:00 | 5024.00 | 5023.30 | 5039.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 12:30:00 | 5021.00 | 5023.30 | 5039.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4977.50 | 5011.82 | 5028.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 4908.00 | 4960.22 | 4987.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 4860.00 | 4834.23 | 4832.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 4860.00 | 4834.23 | 4832.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 4904.00 | 4854.04 | 4842.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 4896.50 | 4898.20 | 4877.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 4937.50 | 4898.20 | 4877.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:15:00 | 4928.50 | 4899.26 | 4879.54 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 4912.50 | 4902.89 | 4886.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 4893.00 | 4902.89 | 4886.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 4898.00 | 4901.91 | 4887.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 4897.00 | 4901.91 | 4887.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 4889.50 | 4905.86 | 4893.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 4889.50 | 4905.86 | 4893.38 | SL hit (close<ema400) qty=1.00 sl=4893.38 alert=retest1 |

### Cycle 88 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 4869.50 | 4885.79 | 4886.13 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 4957.50 | 4894.34 | 4888.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 12:15:00 | 5056.50 | 4926.77 | 4903.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 5097.00 | 5114.22 | 5071.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:15:00 | 5096.50 | 5114.22 | 5071.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 5103.50 | 5110.44 | 5077.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:45:00 | 5064.00 | 5110.44 | 5077.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 5060.00 | 5100.35 | 5075.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 5060.00 | 5100.35 | 5075.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 5041.00 | 5088.48 | 5072.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 5041.00 | 5088.48 | 5072.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 5072.50 | 5082.57 | 5072.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 5114.50 | 5082.57 | 5072.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 5076.00 | 5137.18 | 5136.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 5074.50 | 5124.64 | 5131.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 5074.50 | 5124.64 | 5131.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 5067.00 | 5105.41 | 5120.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 5058.50 | 5049.72 | 5078.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:30:00 | 5061.00 | 5049.72 | 5078.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 5080.00 | 5056.38 | 5076.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 4985.00 | 5060.11 | 5076.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 5038.00 | 5021.67 | 5030.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 5042.00 | 5025.73 | 5031.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 5044.00 | 5030.67 | 5033.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 5060.00 | 5036.54 | 5035.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 5060.00 | 5036.54 | 5035.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 5090.00 | 5052.69 | 5043.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 5070.00 | 5078.66 | 5060.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:00:00 | 5070.00 | 5078.66 | 5060.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 5065.50 | 5076.03 | 5061.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 5065.00 | 5076.03 | 5061.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 5057.00 | 5072.23 | 5060.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 5062.00 | 5072.23 | 5060.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 5031.00 | 5063.98 | 5058.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 5031.00 | 5063.98 | 5058.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 5034.00 | 5057.98 | 5056.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 5062.50 | 5057.98 | 5056.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 5131.50 | 5167.57 | 5171.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 5131.50 | 5167.57 | 5171.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 5100.00 | 5154.06 | 5165.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 4520.00 | 4496.83 | 4557.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:30:00 | 4482.10 | 4496.83 | 4557.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 4628.00 | 4526.29 | 4560.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 4628.00 | 4526.29 | 4560.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 4621.20 | 4545.28 | 4566.42 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 4650.00 | 4582.64 | 4580.76 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 4536.70 | 4571.49 | 4575.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 4515.60 | 4560.31 | 4570.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 4606.70 | 4545.34 | 4556.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 4606.70 | 4545.34 | 4556.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4606.70 | 4545.34 | 4556.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 4609.00 | 4545.34 | 4556.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 4583.90 | 4553.05 | 4558.55 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 11:15:00 | 4626.60 | 4567.76 | 4564.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 4657.00 | 4608.28 | 4586.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 4557.40 | 4598.10 | 4583.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 4557.40 | 4598.10 | 4583.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 4557.40 | 4598.10 | 4583.97 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 4523.10 | 4571.65 | 4573.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 4494.20 | 4528.62 | 4544.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 4541.90 | 4516.25 | 4529.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 4541.90 | 4516.25 | 4529.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 4541.90 | 4516.25 | 4529.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 4556.70 | 4516.25 | 4529.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 4510.00 | 4515.00 | 4527.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 4482.50 | 4504.46 | 4518.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 4539.20 | 4522.11 | 4521.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 4539.20 | 4522.11 | 4521.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 4763.90 | 4572.54 | 4544.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 5112.30 | 5214.94 | 5146.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 5112.30 | 5214.94 | 5146.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 5112.30 | 5214.94 | 5146.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 5125.00 | 5214.94 | 5146.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 5079.30 | 5187.82 | 5139.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 5079.30 | 5187.82 | 5139.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 5030.00 | 5113.88 | 5114.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 4963.00 | 5083.71 | 5100.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 10:15:00 | 5100.00 | 5061.26 | 5083.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 10:15:00 | 5100.00 | 5061.26 | 5083.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 5100.00 | 5061.26 | 5083.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 5100.00 | 5061.26 | 5083.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 5047.20 | 5058.45 | 5080.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 5035.10 | 5058.45 | 5080.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 4783.35 | 4966.24 | 5027.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-29 09:15:00 | 4531.59 | 4670.62 | 4812.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 4514.30 | 4472.61 | 4470.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 4532.20 | 4494.44 | 4482.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 15:15:00 | 4500.30 | 4504.20 | 4489.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 15:15:00 | 4500.30 | 4504.20 | 4489.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 4500.30 | 4504.20 | 4489.92 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 4462.90 | 4481.35 | 4482.84 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 4514.80 | 4488.80 | 4485.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 4573.90 | 4512.89 | 4498.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 4561.20 | 4561.96 | 4535.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 4561.20 | 4561.96 | 4535.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 4525.70 | 4553.45 | 4536.31 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 4494.70 | 4524.51 | 4527.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 4475.00 | 4505.11 | 4515.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 4525.90 | 4457.19 | 4472.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 4525.90 | 4457.19 | 4472.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 4525.90 | 4457.19 | 4472.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 4515.00 | 4457.19 | 4472.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 4518.10 | 4469.37 | 4476.25 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 4524.20 | 4487.29 | 4483.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 4526.90 | 4495.21 | 4487.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 4597.60 | 4613.18 | 4573.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 4597.60 | 4613.18 | 4573.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 4597.60 | 4613.18 | 4573.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 4597.60 | 4613.18 | 4573.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 4590.60 | 4601.97 | 4580.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 4594.30 | 4601.97 | 4580.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 4585.40 | 4598.66 | 4580.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 4588.60 | 4598.66 | 4580.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 4590.00 | 4596.93 | 4581.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 4588.40 | 4596.93 | 4581.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 4593.90 | 4596.32 | 4582.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 4630.00 | 4596.32 | 4582.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 4585.00 | 4594.06 | 4582.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 4584.70 | 4594.06 | 4582.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 4531.70 | 4581.59 | 4578.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 4531.70 | 4581.59 | 4578.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 4530.80 | 4571.43 | 4574.00 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 4592.50 | 4575.94 | 4575.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 4645.50 | 4596.00 | 4585.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 11:15:00 | 4619.50 | 4635.02 | 4613.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 11:15:00 | 4619.50 | 4635.02 | 4613.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 4619.50 | 4635.02 | 4613.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 4591.70 | 4635.02 | 4613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 4642.00 | 4638.71 | 4622.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:45:00 | 4675.00 | 4643.89 | 4626.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 4655.60 | 4648.39 | 4633.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 4669.90 | 4647.60 | 4637.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 4613.80 | 4629.77 | 4631.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 4613.80 | 4629.77 | 4631.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 4577.80 | 4619.37 | 4626.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 4428.90 | 4401.58 | 4464.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 11:00:00 | 4428.90 | 4401.58 | 4464.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 4470.90 | 4415.83 | 4460.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 4470.90 | 4415.83 | 4460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 4459.00 | 4424.47 | 4460.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 4459.00 | 4424.47 | 4460.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 4460.10 | 4431.59 | 4460.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 4460.10 | 4431.59 | 4460.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 4230.00 | 4391.27 | 4439.32 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 4461.60 | 4415.68 | 4414.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 4478.00 | 4449.03 | 4433.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 4441.50 | 4456.23 | 4441.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 4441.50 | 4456.23 | 4441.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 4441.50 | 4456.23 | 4441.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 4441.50 | 4456.23 | 4441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 4434.50 | 4451.88 | 4440.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 4434.50 | 4451.88 | 4440.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 4416.70 | 4444.85 | 4438.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 4416.70 | 4444.85 | 4438.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 4421.50 | 4440.18 | 4436.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 4417.10 | 4440.18 | 4436.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 4419.70 | 4432.02 | 4433.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 4395.80 | 4424.78 | 4430.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 4371.00 | 4365.05 | 4385.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 4371.00 | 4365.05 | 4385.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 4391.60 | 4370.36 | 4386.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:30:00 | 4408.60 | 4370.36 | 4386.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 4390.00 | 4374.29 | 4386.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 4408.80 | 4379.99 | 4388.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 4421.00 | 4388.19 | 4391.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 4421.00 | 4388.19 | 4391.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 4413.60 | 4393.28 | 4393.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 4423.90 | 4399.40 | 4396.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 15:15:00 | 4398.90 | 4404.00 | 4399.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 15:15:00 | 4398.90 | 4404.00 | 4399.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 4398.90 | 4404.00 | 4399.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 4448.70 | 4404.00 | 4399.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4419.00 | 4407.00 | 4401.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 4482.90 | 4427.23 | 4418.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 4465.00 | 4433.89 | 4423.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 4466.70 | 4433.89 | 4423.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 4465.40 | 4438.25 | 4426.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 4483.00 | 4489.96 | 4467.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 4480.10 | 4489.96 | 4467.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 4494.00 | 4491.52 | 4475.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 4455.70 | 4473.15 | 4474.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 4455.70 | 4473.15 | 4474.42 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 4516.10 | 4481.74 | 4478.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 4541.20 | 4499.75 | 4487.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 4651.00 | 4662.11 | 4631.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 4711.20 | 4670.91 | 4638.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 4626.80 | 4677.41 | 4661.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 4626.80 | 4677.41 | 4661.53 | SL hit (close<ema400) qty=1.00 sl=4661.53 alert=retest1 |

### Cycle 112 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 4635.20 | 4649.47 | 4651.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 4616.30 | 4640.55 | 4646.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 4601.10 | 4597.04 | 4617.34 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 15:15:00 | 4542.00 | 4598.78 | 4610.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 4542.00 | 4587.42 | 4604.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 4614.90 | 4592.92 | 4605.39 | SL hit (close>ema400) qty=1.00 sl=4605.39 alert=retest1 |

### Cycle 113 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 4692.50 | 4623.83 | 4616.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 4761.40 | 4668.72 | 4640.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 4817.20 | 4823.62 | 4785.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:00:00 | 4817.20 | 4823.62 | 4785.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 4800.00 | 4812.96 | 4789.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:15:00 | 4815.00 | 4812.96 | 4789.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 4826.10 | 4805.53 | 4795.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 4778.10 | 4797.26 | 4796.32 | SL hit (close<static) qty=1.00 sl=4784.80 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 4623.20 | 4762.45 | 4780.58 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 4779.60 | 4745.97 | 4744.92 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 4626.00 | 4734.27 | 4742.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 4619.70 | 4711.35 | 4731.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 4710.60 | 4647.90 | 4670.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 14:15:00 | 4710.60 | 4647.90 | 4670.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 4710.60 | 4647.90 | 4670.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:45:00 | 4703.40 | 4647.90 | 4670.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 4702.00 | 4658.72 | 4672.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 4720.50 | 4658.72 | 4672.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 4736.40 | 4685.66 | 4683.46 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 4661.90 | 4680.08 | 4681.69 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 4711.00 | 4681.31 | 4681.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 4724.00 | 4689.85 | 4685.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 09:15:00 | 4673.60 | 4692.87 | 4689.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 4673.60 | 4692.87 | 4689.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 4673.60 | 4692.87 | 4689.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 4673.60 | 4692.87 | 4689.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 4729.70 | 4700.23 | 4692.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 4747.40 | 4710.20 | 4699.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 4739.40 | 4710.20 | 4699.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 4737.40 | 4714.76 | 4702.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 4740.90 | 4716.97 | 4704.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 4691.20 | 4713.97 | 4706.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 4691.20 | 4713.97 | 4706.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 4734.80 | 4718.13 | 4708.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 4745.00 | 4723.51 | 4712.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 15:00:00 | 4757.50 | 4733.12 | 4718.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 4759.30 | 4733.10 | 4719.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:45:00 | 4755.80 | 4749.18 | 4734.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 4732.80 | 4745.91 | 4734.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 4732.80 | 4745.91 | 4734.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 4720.00 | 4740.73 | 4732.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 4695.00 | 4740.73 | 4732.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 4670.80 | 4726.74 | 4727.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 4670.80 | 4726.74 | 4727.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 4645.40 | 4679.26 | 4698.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 4671.80 | 4668.75 | 4686.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 4671.80 | 4668.75 | 4686.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 4732.90 | 4681.58 | 4690.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 4732.90 | 4681.58 | 4690.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 4720.00 | 4689.27 | 4693.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 4733.70 | 4695.21 | 4695.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 4740.00 | 4704.17 | 4699.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 14:15:00 | 4765.80 | 4730.57 | 4714.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 4691.70 | 4725.91 | 4715.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 4691.70 | 4725.91 | 4715.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 4691.70 | 4725.91 | 4715.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 4685.20 | 4725.91 | 4715.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 4687.70 | 4718.27 | 4713.17 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 4685.70 | 4708.46 | 4709.39 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 4749.60 | 4716.90 | 4713.08 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 4665.00 | 4705.42 | 4708.46 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 4853.10 | 4715.45 | 4702.48 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 13:15:00 | 4690.00 | 4727.40 | 4730.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 4674.60 | 4699.41 | 4713.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 15:15:00 | 4740.00 | 4705.48 | 4713.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 4740.00 | 4705.48 | 4713.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 4740.00 | 4705.48 | 4713.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 4670.00 | 4694.95 | 4707.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 4794.50 | 4722.77 | 4715.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 4794.50 | 4722.77 | 4715.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 4821.50 | 4753.60 | 4731.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 4751.00 | 4764.15 | 4743.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 4756.40 | 4764.15 | 4743.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 4762.20 | 4763.76 | 4744.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 4782.60 | 4763.76 | 4744.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 4773.60 | 4772.00 | 4769.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 4732.50 | 4764.10 | 4765.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 4732.50 | 4764.10 | 4765.86 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 4779.70 | 4768.99 | 4767.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 4787.00 | 4774.34 | 4770.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 4780.00 | 4793.17 | 4785.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 4780.00 | 4793.17 | 4785.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4780.00 | 4793.17 | 4785.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 4787.90 | 4793.17 | 4785.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 4809.50 | 4796.43 | 4787.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 4809.50 | 4796.43 | 4787.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4785.10 | 4794.91 | 4788.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 4847.00 | 4799.19 | 4791.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-01 14:15:00 | 5331.70 | 5294.46 | 5261.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 5298.50 | 5311.38 | 5312.63 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 5325.00 | 5314.10 | 5313.76 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 5245.00 | 5300.28 | 5307.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 5177.00 | 5275.62 | 5295.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 5296.50 | 5271.41 | 5285.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 5296.50 | 5271.41 | 5285.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 5296.50 | 5271.41 | 5285.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 5283.00 | 5271.41 | 5285.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 5274.00 | 5271.93 | 5284.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 5230.00 | 5264.88 | 5276.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 5241.00 | 5257.38 | 5269.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 5250.00 | 5254.92 | 5266.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 5242.50 | 5256.77 | 5265.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5255.00 | 5253.41 | 5262.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 5191.00 | 5235.32 | 5252.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4968.50 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4978.95 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4987.50 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4980.38 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4931.45 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 5055.00 | 5033.55 | 5092.32 | SL hit (close>ema200) qty=0.50 sl=5033.55 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 5214.00 | 5122.92 | 5117.06 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 5061.50 | 5115.10 | 5120.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 5022.50 | 5079.15 | 5100.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 5090.00 | 5066.59 | 5086.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 5090.00 | 5066.59 | 5086.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 5090.00 | 5066.59 | 5086.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 5090.00 | 5066.59 | 5086.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 5048.00 | 5062.88 | 5082.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 15:15:00 | 5010.00 | 5060.00 | 5079.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 5012.50 | 5037.40 | 5065.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 5094.00 | 5055.32 | 5056.16 | SL hit (close>static) qty=1.00 sl=5090.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 5070.00 | 5058.25 | 5057.42 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 5024.00 | 5056.77 | 5057.44 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 5158.00 | 5075.93 | 5065.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 10:15:00 | 5198.50 | 5100.45 | 5078.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 15:15:00 | 5153.50 | 5158.44 | 5120.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 5212.50 | 5169.25 | 5128.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 15:00:00 | 5240.00 | 5193.21 | 5156.27 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 11:00:00 | 5217.50 | 5204.02 | 5171.25 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 5182.00 | 5200.17 | 5177.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:30:00 | 5175.00 | 5200.17 | 5177.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 5173.00 | 5194.74 | 5177.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 5173.00 | 5194.74 | 5177.32 | SL hit (close<ema400) qty=1.00 sl=5177.32 alert=retest1 |

### Cycle 138 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 5151.00 | 5167.65 | 5168.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 5133.50 | 5160.82 | 5165.44 | Break + close below crossover candle low |

### Cycle 139 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 5240.00 | 5176.66 | 5172.21 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 5075.00 | 5159.92 | 5171.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 4980.00 | 5123.94 | 5153.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 5088.50 | 5066.63 | 5107.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:30:00 | 5083.00 | 5066.63 | 5107.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 5181.50 | 5089.60 | 5113.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 5181.50 | 5089.60 | 5113.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 5370.50 | 5145.78 | 5137.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 5399.50 | 5196.53 | 5161.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 5357.00 | 5374.08 | 5320.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 5365.00 | 5374.08 | 5320.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 5337.00 | 5366.66 | 5322.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:30:00 | 5406.50 | 5357.82 | 5335.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:00:00 | 5443.50 | 5388.39 | 5361.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 13:15:00 | 5344.00 | 5390.21 | 5396.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 5344.00 | 5390.21 | 5396.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 5322.00 | 5376.57 | 5389.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 5343.00 | 5342.44 | 5368.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 5343.00 | 5342.44 | 5368.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 5343.00 | 5342.44 | 5368.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 5350.00 | 5342.44 | 5368.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 5360.00 | 5335.41 | 5355.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 5360.00 | 5335.41 | 5355.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 5372.50 | 5342.83 | 5357.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 5283.50 | 5342.83 | 5357.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 5492.50 | 5379.60 | 5367.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 5492.50 | 5379.60 | 5367.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 15:15:00 | 5550.00 | 5413.68 | 5383.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 5281.00 | 5387.15 | 5374.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 5281.00 | 5387.15 | 5374.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 5281.00 | 5387.15 | 5374.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 5283.50 | 5387.15 | 5374.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 5270.50 | 5363.82 | 5365.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 09:15:00 | 5248.00 | 5319.64 | 5335.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 10:15:00 | 5204.00 | 5200.73 | 5233.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 5230.00 | 5202.41 | 5221.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 5230.00 | 5202.41 | 5221.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 5150.00 | 5202.41 | 5221.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 5135.00 | 5188.93 | 5213.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 5231.00 | 5201.84 | 5215.65 | SL hit (close>static) qty=1.00 sl=5230.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 5379.50 | 5240.18 | 5225.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 5485.00 | 5308.85 | 5267.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 5350.00 | 5350.53 | 5304.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 5350.00 | 5350.53 | 5304.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 5470.00 | 5465.93 | 5417.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 5491.50 | 5465.93 | 5417.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 5482.50 | 5469.24 | 5423.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:30:00 | 5499.00 | 5478.80 | 5432.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 5329.00 | 5463.52 | 5445.57 | SL hit (close<static) qty=1.00 sl=5391.50 alert=retest2 |

### Cycle 146 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 5256.50 | 5406.27 | 5421.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5216.00 | 5368.22 | 5402.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 5315.00 | 5278.66 | 5328.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 5315.00 | 5278.66 | 5328.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 5478.50 | 5318.63 | 5342.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 5478.50 | 5318.63 | 5342.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 5428.00 | 5340.50 | 5350.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 5397.00 | 5340.50 | 5350.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 5498.00 | 5381.04 | 5367.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 5498.00 | 5381.04 | 5367.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 5590.00 | 5467.26 | 5418.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 5443.00 | 5484.24 | 5440.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 5443.00 | 5484.24 | 5440.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 5443.00 | 5484.24 | 5440.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 5443.00 | 5484.24 | 5440.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 5462.00 | 5479.79 | 5442.61 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 5349.50 | 5418.47 | 5422.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 5231.00 | 5380.97 | 5405.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 5480.00 | 5315.77 | 5352.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 5480.00 | 5315.77 | 5352.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 5480.00 | 5315.77 | 5352.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 5480.00 | 5315.77 | 5352.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 5410.00 | 5334.62 | 5357.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 5419.50 | 5353.59 | 5364.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5505.50 | 5383.98 | 5377.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 5517.50 | 5410.68 | 5390.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 15:15:00 | 5445.00 | 5446.07 | 5416.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 09:15:00 | 5429.50 | 5446.07 | 5416.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 5431.50 | 5443.15 | 5418.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 5396.50 | 5443.15 | 5418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 5373.00 | 5429.12 | 5414.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 5373.00 | 5429.12 | 5414.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 5362.00 | 5415.70 | 5409.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 5370.50 | 5415.70 | 5409.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 5357.50 | 5404.06 | 5404.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 5316.50 | 5386.55 | 5396.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 5339.50 | 5338.80 | 5363.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:30:00 | 5335.50 | 5338.80 | 5363.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 5345.50 | 5338.66 | 5358.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 5345.50 | 5338.66 | 5358.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 5301.00 | 5331.12 | 5353.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 5270.50 | 5331.12 | 5353.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:15:00 | 5006.97 | 5066.22 | 5150.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 5020.50 | 5014.51 | 5094.32 | SL hit (close>ema200) qty=0.50 sl=5014.51 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 4750.50 | 4688.53 | 4688.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 4793.00 | 4720.94 | 4703.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4617.00 | 4720.84 | 4711.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4617.00 | 4720.84 | 4711.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4617.00 | 4720.84 | 4711.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 4617.00 | 4720.84 | 4711.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 4606.00 | 4697.87 | 4701.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 4592.50 | 4676.80 | 4691.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4467.00 | 4462.84 | 4538.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 10:15:00 | 4538.30 | 4477.93 | 4538.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4538.30 | 4477.93 | 4538.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 4538.30 | 4477.93 | 4538.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 4535.20 | 4489.38 | 4538.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 4535.20 | 4489.38 | 4538.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 4579.90 | 4507.49 | 4541.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 4579.90 | 4507.49 | 4541.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 4564.30 | 4518.85 | 4543.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 4470.50 | 4541.68 | 4550.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 4516.50 | 4507.27 | 4522.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 4519.30 | 4521.61 | 4527.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 4521.80 | 4521.65 | 4526.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 4513.00 | 4519.92 | 4525.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 4513.00 | 4519.92 | 4525.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 4522.10 | 4512.77 | 4520.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 4522.10 | 4512.77 | 4520.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 4591.70 | 4528.55 | 4526.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 4591.70 | 4528.55 | 4526.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 4683.10 | 4597.12 | 4565.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 4804.30 | 4831.52 | 4738.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 4804.30 | 4831.52 | 4738.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4830.00 | 4904.74 | 4865.73 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 4801.60 | 4839.81 | 4844.26 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 4922.00 | 4854.12 | 4846.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 5040.00 | 4898.59 | 4868.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 5000.00 | 5008.51 | 4986.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 5011.30 | 5008.51 | 4986.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 5019.30 | 5010.67 | 4989.53 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 4916.30 | 4973.57 | 4979.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 4909.10 | 4960.68 | 4972.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 4952.30 | 4950.91 | 4964.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 14:00:00 | 4952.30 | 4950.91 | 4964.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4611.00 | 4561.47 | 4597.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 4617.40 | 4561.47 | 4597.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 4637.30 | 4576.64 | 4601.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 4637.30 | 4576.64 | 4601.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 4641.10 | 4589.53 | 4605.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 4641.10 | 4589.53 | 4605.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 4601.60 | 4565.25 | 4579.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 4601.60 | 4565.25 | 4579.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 4626.00 | 4577.40 | 4583.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 4611.60 | 4577.40 | 4583.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 4669.80 | 4603.08 | 4594.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 4678.50 | 4636.39 | 4613.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 4712.70 | 4723.63 | 4685.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 4733.50 | 4723.63 | 4685.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 4703.30 | 4723.26 | 4700.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 4702.30 | 4723.26 | 4700.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 4722.00 | 4723.00 | 4702.26 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 09:30:00 | 4759.80 | 2024-05-16 13:15:00 | 4595.05 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-05-15 11:45:00 | 4749.00 | 2024-05-16 13:15:00 | 4595.05 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-05-15 13:15:00 | 4748.45 | 2024-05-16 13:15:00 | 4595.05 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-05-15 15:00:00 | 4743.60 | 2024-05-16 13:15:00 | 4595.05 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-05-23 12:00:00 | 4543.10 | 2024-05-28 09:15:00 | 4315.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 11:00:00 | 4546.25 | 2024-05-28 09:15:00 | 4318.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 11:45:00 | 4552.30 | 2024-05-28 09:15:00 | 4324.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 13:30:00 | 4532.35 | 2024-05-28 09:15:00 | 4305.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 12:00:00 | 4543.10 | 2024-05-30 11:15:00 | 4248.00 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2024-05-24 11:00:00 | 4546.25 | 2024-05-30 11:15:00 | 4248.00 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2024-05-24 11:45:00 | 4552.30 | 2024-05-30 11:15:00 | 4248.00 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2024-05-24 13:30:00 | 4532.35 | 2024-05-30 11:15:00 | 4248.00 | STOP_HIT | 0.50 | 6.27% |
| SELL | retest2 | 2024-05-28 09:15:00 | 4309.70 | 2024-05-30 14:15:00 | 4094.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:15:00 | 4309.70 | 2024-05-31 10:15:00 | 4223.50 | STOP_HIT | 0.50 | 2.00% |
| BUY | retest2 | 2024-06-27 09:15:00 | 4890.00 | 2024-06-27 12:15:00 | 4781.75 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-07-04 11:30:00 | 4593.05 | 2024-07-05 10:15:00 | 4601.80 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-07-04 12:00:00 | 4583.25 | 2024-07-05 10:15:00 | 4601.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-07-04 13:15:00 | 4591.10 | 2024-07-05 10:15:00 | 4601.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-07-04 13:45:00 | 4591.50 | 2024-07-05 10:15:00 | 4601.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-11 09:15:00 | 4484.35 | 2024-07-15 11:15:00 | 4260.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 09:15:00 | 4484.35 | 2024-07-18 09:15:00 | 4286.95 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2024-07-31 09:15:00 | 4306.00 | 2024-08-05 09:15:00 | 4090.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 10:30:00 | 4324.85 | 2024-08-05 09:15:00 | 4108.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 10:30:00 | 4333.00 | 2024-08-05 09:15:00 | 4116.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 12:45:00 | 4305.50 | 2024-08-05 09:15:00 | 4090.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 09:15:00 | 4306.00 | 2024-08-06 13:15:00 | 3892.37 | TARGET_HIT | 0.50 | 9.61% |
| SELL | retest2 | 2024-08-01 10:30:00 | 4324.85 | 2024-08-06 13:15:00 | 3899.70 | TARGET_HIT | 0.50 | 9.83% |
| SELL | retest2 | 2024-08-02 10:30:00 | 4333.00 | 2024-08-06 14:15:00 | 3875.40 | TARGET_HIT | 0.50 | 10.56% |
| SELL | retest2 | 2024-08-02 12:45:00 | 4305.50 | 2024-08-06 14:15:00 | 3874.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-06 09:30:00 | 3992.95 | 2024-08-08 10:15:00 | 3793.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 09:30:00 | 3992.95 | 2024-08-09 09:15:00 | 3845.15 | STOP_HIT | 0.50 | 3.70% |
| BUY | retest2 | 2024-08-26 09:15:00 | 4328.05 | 2024-08-29 15:15:00 | 4760.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-16 09:15:00 | 4927.00 | 2024-09-23 13:15:00 | 4940.20 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-09-16 14:45:00 | 4866.95 | 2024-09-23 13:15:00 | 4940.20 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2024-10-08 09:15:00 | 5413.65 | 2024-10-10 09:15:00 | 5204.40 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2024-10-08 14:00:00 | 5369.55 | 2024-10-10 09:15:00 | 5204.40 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-10-09 09:45:00 | 5374.25 | 2024-10-10 09:15:00 | 5204.40 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-10-09 10:30:00 | 5371.05 | 2024-10-10 09:15:00 | 5204.40 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2024-10-14 14:00:00 | 5086.50 | 2024-10-14 15:15:00 | 5160.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-15 09:45:00 | 5119.95 | 2024-10-17 10:15:00 | 4863.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 11:30:00 | 5112.65 | 2024-10-17 14:15:00 | 4857.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:45:00 | 5119.95 | 2024-10-18 13:15:00 | 4855.00 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2024-10-15 11:30:00 | 5112.65 | 2024-10-18 13:15:00 | 4855.00 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2024-10-18 15:15:00 | 4970.00 | 2024-10-21 09:15:00 | 5115.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-10-23 14:15:00 | 4942.70 | 2024-10-24 13:15:00 | 4695.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:15:00 | 4942.70 | 2024-10-25 14:15:00 | 4925.50 | STOP_HIT | 0.50 | 0.35% |
| BUY | retest2 | 2024-11-01 18:00:00 | 4841.25 | 2024-11-06 13:15:00 | 4771.95 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-11-05 09:30:00 | 4972.90 | 2024-11-06 13:15:00 | 4771.95 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2024-11-26 09:15:00 | 4376.50 | 2024-11-26 13:15:00 | 4198.30 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2024-11-26 10:00:00 | 4319.30 | 2024-11-26 13:15:00 | 4198.30 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-12-02 13:30:00 | 4016.85 | 2024-12-03 12:15:00 | 4170.00 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2024-12-02 15:15:00 | 4028.00 | 2024-12-03 12:15:00 | 4170.00 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2024-12-12 11:15:00 | 4275.00 | 2024-12-12 11:15:00 | 4285.40 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-12-18 10:15:00 | 4475.30 | 2024-12-20 13:15:00 | 4344.80 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-12-18 11:00:00 | 4479.85 | 2024-12-20 13:15:00 | 4344.80 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-12-18 15:00:00 | 4477.35 | 2024-12-20 13:15:00 | 4344.80 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-12-19 15:00:00 | 4543.00 | 2024-12-20 13:15:00 | 4344.80 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2024-12-27 12:15:00 | 4326.00 | 2024-12-30 14:15:00 | 4109.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 12:30:00 | 4339.95 | 2024-12-30 14:15:00 | 4122.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 12:15:00 | 4326.00 | 2024-12-31 15:15:00 | 4194.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2024-12-30 12:30:00 | 4339.95 | 2024-12-31 15:15:00 | 4194.00 | STOP_HIT | 0.50 | 3.36% |
| BUY | retest2 | 2025-01-07 10:45:00 | 4493.80 | 2025-01-07 15:15:00 | 4426.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-01-16 09:15:00 | 4656.95 | 2025-01-16 10:15:00 | 4556.85 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-01-16 14:00:00 | 4614.40 | 2025-01-16 14:15:00 | 4472.05 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-01-29 11:15:00 | 4420.40 | 2025-01-31 09:15:00 | 4575.05 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-01-29 11:45:00 | 4420.00 | 2025-01-31 09:15:00 | 4575.05 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-01-30 09:45:00 | 4409.40 | 2025-01-31 09:15:00 | 4575.05 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2025-02-06 09:30:00 | 4644.55 | 2025-02-10 09:15:00 | 4543.55 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-02-07 11:15:00 | 4640.55 | 2025-02-10 09:15:00 | 4543.55 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-02-07 12:15:00 | 4653.20 | 2025-02-10 09:15:00 | 4543.55 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-02-07 15:00:00 | 4665.00 | 2025-02-10 09:15:00 | 4543.55 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-03-03 10:30:00 | 4420.00 | 2025-03-04 11:15:00 | 4550.55 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-03-05 10:15:00 | 4588.20 | 2025-03-06 12:15:00 | 4502.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-03-05 13:15:00 | 4583.50 | 2025-03-06 12:15:00 | 4502.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-03-11 09:15:00 | 4398.85 | 2025-03-11 15:15:00 | 4508.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-03-17 09:15:00 | 4677.00 | 2025-03-24 15:15:00 | 4777.00 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2025-04-04 13:30:00 | 4468.05 | 2025-04-04 14:15:00 | 4571.70 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-04-15 09:15:00 | 4841.70 | 2025-04-22 09:15:00 | 5325.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-06 11:45:00 | 5114.50 | 2025-05-07 09:15:00 | 5223.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-05-06 12:45:00 | 5109.50 | 2025-05-07 09:15:00 | 5223.50 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-05-12 12:00:00 | 5018.00 | 2025-05-14 10:15:00 | 5141.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-05-13 09:15:00 | 5045.00 | 2025-05-14 10:15:00 | 5141.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-05-13 09:45:00 | 5029.50 | 2025-05-14 10:15:00 | 5141.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-23 11:15:00 | 5198.00 | 2025-05-28 09:15:00 | 5114.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-05-26 11:15:00 | 5201.50 | 2025-05-28 09:15:00 | 5114.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-26 14:30:00 | 5344.50 | 2025-05-28 09:15:00 | 5114.00 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2025-05-30 10:30:00 | 5062.00 | 2025-05-30 14:15:00 | 5147.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-30 13:00:00 | 5072.00 | 2025-05-30 14:15:00 | 5147.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-06-04 14:15:00 | 5078.00 | 2025-06-06 14:15:00 | 5075.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-06-04 15:00:00 | 5058.50 | 2025-06-06 14:15:00 | 5075.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-06-06 13:00:00 | 5075.50 | 2025-06-06 14:15:00 | 5075.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-06-17 14:30:00 | 4908.00 | 2025-06-23 11:15:00 | 4860.00 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest1 | 2025-06-25 09:15:00 | 4937.50 | 2025-06-26 09:15:00 | 4889.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2025-06-25 10:15:00 | 4928.50 | 2025-06-26 09:15:00 | 4889.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-03 09:15:00 | 5114.50 | 2025-07-09 10:15:00 | 5074.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-09 10:15:00 | 5076.00 | 2025-07-09 10:15:00 | 5074.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-11 09:15:00 | 4985.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-14 14:15:00 | 5038.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-14 15:00:00 | 5042.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-15 09:30:00 | 5044.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-17 09:15:00 | 5062.50 | 2025-07-24 10:15:00 | 5131.50 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2025-08-13 14:45:00 | 4482.50 | 2025-08-14 13:15:00 | 4539.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-26 12:15:00 | 5035.10 | 2025-08-26 15:15:00 | 4783.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 12:15:00 | 5035.10 | 2025-08-29 09:15:00 | 4531.59 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-24 09:45:00 | 4675.00 | 2025-09-25 13:15:00 | 4613.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-24 12:45:00 | 4655.60 | 2025-09-25 13:15:00 | 4613.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-25 09:30:00 | 4669.90 | 2025-09-25 13:15:00 | 4613.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-10-14 13:45:00 | 4482.90 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-15 09:30:00 | 4465.00 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-10-15 10:00:00 | 4466.70 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-15 11:15:00 | 4465.40 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-28 09:30:00 | 4711.20 | 2025-10-29 09:15:00 | 4626.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest1 | 2025-10-31 15:15:00 | 4542.00 | 2025-11-03 09:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-10 10:15:00 | 4815.00 | 2025-11-11 15:15:00 | 4778.10 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-11 09:15:00 | 4826.10 | 2025-11-11 15:15:00 | 4778.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-24 12:30:00 | 4747.40 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-24 13:15:00 | 4739.40 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-24 13:45:00 | 4737.40 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-11-24 14:45:00 | 4740.90 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-25 13:00:00 | 4745.00 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-11-25 15:00:00 | 4757.50 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-26 09:15:00 | 4759.30 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-11-26 13:45:00 | 4755.80 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-11 10:45:00 | 4670.00 | 2025-12-12 10:15:00 | 4794.50 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-12-15 10:15:00 | 4782.60 | 2025-12-16 15:15:00 | 4732.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-16 15:00:00 | 4773.60 | 2025-12-16 15:15:00 | 4732.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-19 13:15:00 | 4847.00 | 2026-01-01 14:15:00 | 5331.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-09 15:15:00 | 5230.00 | 2026-01-16 09:15:00 | 4968.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 11:15:00 | 5241.00 | 2026-01-16 09:15:00 | 4978.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 13:15:00 | 5250.00 | 2026-01-16 09:15:00 | 4987.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 15:00:00 | 5242.50 | 2026-01-16 09:15:00 | 4980.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 5191.00 | 2026-01-16 09:15:00 | 4931.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 15:15:00 | 5230.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-01-12 11:15:00 | 5241.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2026-01-12 13:15:00 | 5250.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2026-01-12 15:00:00 | 5242.50 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-01-13 11:45:00 | 5191.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2026-01-21 15:15:00 | 5010.00 | 2026-01-23 10:15:00 | 5094.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-22 09:30:00 | 5012.50 | 2026-01-23 10:15:00 | 5094.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2026-01-28 10:00:00 | 5212.50 | 2026-01-29 14:15:00 | 5173.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2026-01-28 15:00:00 | 5240.00 | 2026-01-29 14:15:00 | 5173.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest1 | 2026-01-29 11:00:00 | 5217.50 | 2026-01-29 14:15:00 | 5173.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-30 10:15:00 | 5167.00 | 2026-01-30 11:15:00 | 5151.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-01-30 11:15:00 | 5153.00 | 2026-01-30 11:15:00 | 5151.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-05 14:30:00 | 5406.50 | 2026-02-10 13:15:00 | 5344.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-06 15:00:00 | 5443.50 | 2026-02-10 13:15:00 | 5344.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-12 09:15:00 | 5283.50 | 2026-02-12 14:15:00 | 5492.50 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-02-20 09:15:00 | 5150.00 | 2026-02-20 11:15:00 | 5231.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-20 10:00:00 | 5135.00 | 2026-02-20 11:15:00 | 5231.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-27 10:15:00 | 5491.50 | 2026-03-02 09:15:00 | 5329.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-02-27 11:00:00 | 5482.50 | 2026-03-02 09:15:00 | 5329.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-02-27 11:30:00 | 5499.00 | 2026-03-02 09:15:00 | 5329.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2026-03-04 15:15:00 | 5397.00 | 2026-03-05 09:15:00 | 5498.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-13 09:15:00 | 5270.50 | 2026-03-17 09:15:00 | 5006.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 5270.50 | 2026-03-17 13:15:00 | 5020.50 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2026-04-02 09:15:00 | 4470.50 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-02 14:45:00 | 4516.50 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-06 09:15:00 | 4519.30 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-06 10:00:00 | 4521.80 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -1.55% |
