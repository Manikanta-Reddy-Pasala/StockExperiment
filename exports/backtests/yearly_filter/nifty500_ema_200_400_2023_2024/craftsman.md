# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 8990.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 42
- **Target hits / Stop hits / Partials:** 4 / 49 / 6
- **Avg / median % per leg:** -0.85% / -2.14%
- **Sum % (uncompounded):** -50.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 3 | 6 | 0 | 1.75% | 15.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 3 | 6 | 0 | 1.75% | 15.7% |
| SELL (all) | 50 | 14 | 28.0% | 1 | 43 | 6 | -1.32% | -65.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 14 | 28.0% | 1 | 43 | 6 | -1.32% | -65.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 59 | 17 | 28.8% | 4 | 49 | 6 | -0.85% | -50.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 09:15:00 | 4729.60 | 4978.41 | 4978.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 4676.90 | 4973.00 | 4976.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 15:15:00 | 4153.00 | 4151.34 | 4353.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 09:15:00 | 4091.00 | 4151.34 | 4353.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 4369.00 | 4159.69 | 4343.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 4399.80 | 4159.69 | 4343.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 4369.70 | 4161.78 | 4344.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 11:30:00 | 4352.55 | 4165.60 | 4344.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 13:45:00 | 4338.50 | 4168.86 | 4344.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-01 13:45:00 | 4347.50 | 4181.29 | 4344.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 09:15:00 | 4351.90 | 4185.03 | 4344.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 4376.20 | 4188.58 | 4344.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:00:00 | 4376.20 | 4188.58 | 4344.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 4354.95 | 4190.24 | 4344.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-03 11:15:00 | 4455.00 | 4203.10 | 4346.04 | SL hit (close>static) qty=1.00 sl=4447.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 11:15:00 | 4689.30 | 4363.41 | 4363.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 4767.05 | 4374.19 | 4368.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 5098.45 | 5143.35 | 4888.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 5098.45 | 5143.35 | 4888.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 5963.60 | 6179.47 | 5971.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 5863.00 | 6179.47 | 5971.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 5933.60 | 6177.03 | 5970.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 5917.55 | 6177.03 | 5970.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 5886.40 | 6174.14 | 5970.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:45:00 | 5881.45 | 6174.14 | 5970.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 5046.80 | 5816.63 | 5819.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 4970.45 | 5808.21 | 5814.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 5131.60 | 5093.78 | 5283.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:00:00 | 5131.60 | 5093.78 | 5283.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 5311.35 | 5094.34 | 5270.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 5311.35 | 5094.34 | 5270.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 5220.00 | 5095.59 | 5270.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:30:00 | 5294.95 | 5095.59 | 5270.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 5236.65 | 5096.55 | 5268.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:30:00 | 5231.95 | 5096.55 | 5268.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 5280.55 | 5098.38 | 5268.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 5280.55 | 5098.38 | 5268.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 5344.95 | 5100.84 | 5268.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:45:00 | 5340.90 | 5100.84 | 5268.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 5313.65 | 5102.95 | 5268.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 5256.25 | 5135.79 | 5276.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:45:00 | 5295.35 | 5137.41 | 5276.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:30:00 | 5293.25 | 5139.12 | 5277.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 5287.75 | 5139.12 | 5277.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 5151.10 | 5140.58 | 5276.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 5375.25 | 5158.29 | 5271.56 | SL hit (close>static) qty=1.00 sl=5364.20 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 4750.00 | 4678.46 | 4678.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 4780.15 | 4679.47 | 4678.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 4565.00 | 4682.52 | 4680.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 4565.00 | 4682.52 | 4680.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 4565.00 | 4682.52 | 4680.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 4565.00 | 4682.52 | 4680.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 4603.90 | 4681.74 | 4680.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 4581.35 | 4681.74 | 4680.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 4563.00 | 4677.60 | 4678.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4306.80 | 4673.91 | 4676.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 4623.80 | 4597.78 | 4635.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 4623.80 | 4597.78 | 4635.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 4623.80 | 4597.78 | 4635.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 4623.80 | 4597.78 | 4635.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 4601.00 | 4597.81 | 4634.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:30:00 | 4607.50 | 4597.81 | 4634.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 4650.30 | 4598.33 | 4635.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 12:45:00 | 4589.80 | 4598.34 | 4634.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 13:15:00 | 4587.60 | 4598.34 | 4634.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 15:15:00 | 4582.30 | 4598.30 | 4634.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 4585.00 | 4599.13 | 4633.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 4556.80 | 4598.80 | 4633.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-17 13:15:00 | 4780.00 | 4601.28 | 4633.70 | SL hit (close>static) qty=1.00 sl=4680.10 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 4769.00 | 4661.63 | 4661.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 4888.20 | 4669.93 | 4665.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 4658.30 | 4684.50 | 4673.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 4658.30 | 4684.50 | 4673.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 4658.30 | 4684.50 | 4673.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 4658.30 | 4684.50 | 4673.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 4603.20 | 4683.69 | 4672.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 4603.20 | 4683.69 | 4672.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 4618.00 | 4682.31 | 4672.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:45:00 | 4621.00 | 4682.31 | 4672.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 4598.50 | 4681.47 | 4671.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 4598.50 | 4681.47 | 4671.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 4668.60 | 4680.22 | 4671.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 4640.00 | 4680.22 | 4671.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 4620.00 | 4679.62 | 4671.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 11:00:00 | 4707.00 | 4679.75 | 4671.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 4579.80 | 4682.11 | 4673.80 | SL hit (close<static) qty=1.00 sl=4585.10 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 6936.00 | 7448.31 | 7448.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 11:15:00 | 6855.00 | 7437.44 | 7442.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 7269.50 | 7095.35 | 7233.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 7378.50 | 7098.17 | 7234.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 7350.00 | 7098.17 | 7234.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 7388.50 | 7151.21 | 7250.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 7388.50 | 7151.21 | 7250.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 7708.00 | 7326.60 | 7326.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7411.62 | 7372.30 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-13 11:30:00 | 4612.95 | 2023-10-09 12:15:00 | 4489.95 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2023-09-13 15:15:00 | 4604.25 | 2023-10-09 12:15:00 | 4489.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-09-22 09:30:00 | 4620.45 | 2023-10-09 12:15:00 | 4489.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2023-10-04 10:45:00 | 4611.05 | 2023-10-09 12:15:00 | 4489.95 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2023-10-13 09:15:00 | 4595.30 | 2023-10-13 09:15:00 | 4551.35 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-10-27 14:15:00 | 4609.85 | 2023-11-16 13:15:00 | 5070.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-28 11:30:00 | 4352.55 | 2024-04-03 11:15:00 | 4455.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-03-28 13:45:00 | 4338.50 | 2024-04-03 11:15:00 | 4455.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-04-01 13:45:00 | 4347.50 | 2024-04-03 11:15:00 | 4455.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-04-02 09:15:00 | 4351.90 | 2024-04-03 11:15:00 | 4455.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-04-15 09:15:00 | 4320.95 | 2024-04-15 09:15:00 | 4504.40 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-04-15 14:15:00 | 4327.15 | 2024-04-15 15:15:00 | 4385.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-04-15 14:45:00 | 4341.60 | 2024-04-15 15:15:00 | 4385.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-04-16 10:00:00 | 4343.80 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-04-19 09:15:00 | 4320.50 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2024-04-19 10:15:00 | 4314.20 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2024-04-19 13:00:00 | 4311.75 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2024-04-19 14:15:00 | 4315.15 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2024-04-22 13:00:00 | 4295.35 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-04-22 14:30:00 | 4300.00 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2024-04-23 12:30:00 | 4305.00 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2024-04-23 13:15:00 | 4289.85 | 2024-04-24 09:15:00 | 4483.40 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2024-04-29 09:15:00 | 4459.05 | 2024-05-03 14:15:00 | 4404.75 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2024-04-29 09:45:00 | 4456.55 | 2024-05-14 10:15:00 | 4422.80 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2024-04-29 10:45:00 | 4453.05 | 2024-05-14 10:15:00 | 4422.80 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2024-05-03 09:15:00 | 4455.75 | 2024-05-14 10:15:00 | 4422.80 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-05-03 13:30:00 | 4359.05 | 2024-05-15 09:15:00 | 4430.35 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-05-06 09:30:00 | 4355.45 | 2024-05-17 10:15:00 | 4424.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-05-07 10:00:00 | 4363.55 | 2024-05-17 10:15:00 | 4424.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-05-14 09:30:00 | 4368.95 | 2024-05-17 10:15:00 | 4424.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-05-14 12:45:00 | 4349.50 | 2024-05-24 15:15:00 | 4450.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-05-15 12:00:00 | 4356.95 | 2024-05-24 15:15:00 | 4450.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-05-15 14:00:00 | 4357.75 | 2024-05-28 13:15:00 | 4236.10 | PARTIAL | 0.50 | 2.79% |
| SELL | retest2 | 2024-05-16 14:00:00 | 4351.00 | 2024-05-28 13:15:00 | 4233.72 | PARTIAL | 0.50 | 2.70% |
| SELL | retest2 | 2024-05-23 14:45:00 | 4295.00 | 2024-05-28 13:15:00 | 4230.40 | PARTIAL | 0.50 | 1.50% |
| SELL | retest2 | 2024-05-24 10:45:00 | 4290.00 | 2024-05-28 13:15:00 | 4232.96 | PARTIAL | 0.50 | 1.33% |
| SELL | retest2 | 2024-05-15 14:00:00 | 4357.75 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-05-16 14:00:00 | 4351.00 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 0.50 | 0.92% |
| SELL | retest2 | 2024-05-23 14:45:00 | 4295.00 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 0.50 | -0.37% |
| SELL | retest2 | 2024-05-24 10:45:00 | 4290.00 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 0.50 | -0.49% |
| SELL | retest2 | 2024-05-28 11:15:00 | 4274.95 | 2024-06-10 09:15:00 | 4441.70 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2024-06-04 09:15:00 | 4130.00 | 2024-06-10 09:15:00 | 4441.70 | STOP_HIT | 1.00 | -7.55% |
| SELL | retest2 | 2024-06-07 12:15:00 | 4322.25 | 2024-06-10 09:15:00 | 4441.70 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-06-07 12:45:00 | 4324.30 | 2024-06-10 09:15:00 | 4441.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-12-20 09:30:00 | 5256.25 | 2024-12-26 14:15:00 | 5375.25 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-12-20 10:45:00 | 5295.35 | 2024-12-26 14:15:00 | 5375.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-12-20 11:30:00 | 5293.25 | 2024-12-26 14:15:00 | 5375.25 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-12-20 12:15:00 | 5287.75 | 2024-12-26 14:15:00 | 5375.25 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-01-13 09:30:00 | 5088.30 | 2025-01-15 09:15:00 | 4833.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 09:30:00 | 5088.30 | 2025-01-22 09:15:00 | 4579.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-05 15:15:00 | 4990.00 | 2025-03-13 14:15:00 | 4740.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-05 15:15:00 | 4990.00 | 2025-03-13 14:15:00 | 4746.80 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-04-15 12:45:00 | 4589.80 | 2025-04-17 13:15:00 | 4780.00 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-04-15 13:15:00 | 4587.60 | 2025-04-17 13:15:00 | 4780.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-04-15 15:15:00 | 4582.30 | 2025-04-17 13:15:00 | 4780.00 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2025-04-16 13:15:00 | 4585.00 | 2025-04-17 13:15:00 | 4780.00 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-05-05 11:00:00 | 4707.00 | 2025-05-08 14:15:00 | 4579.80 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-05-09 10:30:00 | 4792.10 | 2025-05-12 09:15:00 | 5271.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 12:45:00 | 4813.50 | 2025-05-14 09:15:00 | 5294.85 | TARGET_HIT | 1.00 | 10.00% |
