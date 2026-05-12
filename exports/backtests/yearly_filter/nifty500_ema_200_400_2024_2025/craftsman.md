# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 8990.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 20
- **Target hits / Stop hits / Partials:** 3 / 21 / 4
- **Avg / median % per leg:** -0.01% / -1.37%
- **Sum % (uncompounded):** -0.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 2 | 1 | 0 | 5.77% | 17.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 2 | 1 | 0 | 5.77% | 17.3% |
| SELL (all) | 25 | 6 | 24.0% | 1 | 20 | 4 | -0.70% | -17.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 6 | 24.0% | 1 | 20 | 4 | -0.70% | -17.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 8 | 28.6% | 3 | 21 | 4 | -0.01% | -0.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 11:15:00 | 4689.30 | 4363.41 | 4363.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 4767.05 | 4374.19 | 4368.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 5098.45 | 5143.35 | 4888.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 5098.45 | 5143.35 | 4888.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 5963.60 | 6179.47 | 5971.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 5863.00 | 6179.47 | 5971.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 5933.60 | 6177.03 | 5970.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 5917.55 | 6177.03 | 5970.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 5886.40 | 6174.14 | 5970.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:45:00 | 5881.45 | 6174.14 | 5970.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 5046.80 | 5816.63 | 5819.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 4970.45 | 5808.21 | 5814.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 5131.60 | 5093.78 | 5283.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:00:00 | 5131.60 | 5093.78 | 5283.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 5311.35 | 5094.34 | 5270.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 5311.35 | 5094.34 | 5270.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
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

### Cycle 3 — BUY (started 2025-04-02 12:15:00)

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

### Cycle 4 — SELL (started 2025-04-04 15:15:00)

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

### Cycle 5 — BUY (started 2025-04-25 15:15:00)

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

### Cycle 6 — SELL (started 2026-03-18 09:15:00)

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

### Cycle 7 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 7708.00 | 7326.60 | 7326.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7411.62 | 7372.30 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 10:00:00 | 4430.35 | 2024-05-18 09:15:00 | 4450.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-05-15 11:45:00 | 4360.90 | 2024-05-18 09:15:00 | 4450.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-05-16 13:45:00 | 4390.00 | 2024-05-18 09:15:00 | 4450.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-05-17 10:15:00 | 4401.55 | 2024-05-18 09:15:00 | 4450.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-05-17 12:00:00 | 4409.40 | 2024-05-24 15:15:00 | 4450.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-05-23 14:45:00 | 4295.00 | 2024-05-24 15:15:00 | 4450.00 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-05-24 10:45:00 | 4290.00 | 2024-05-28 13:15:00 | 4236.10 | PARTIAL | 0.50 | 1.26% |
| SELL | retest2 | 2024-05-28 11:15:00 | 4274.95 | 2024-05-28 13:15:00 | 4233.72 | PARTIAL | 0.50 | 0.96% |
| SELL | retest2 | 2024-05-24 10:45:00 | 4290.00 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 0.50 | -0.49% |
| SELL | retest2 | 2024-05-28 11:15:00 | 4274.95 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2024-06-04 09:15:00 | 4130.00 | 2024-06-06 10:15:00 | 4311.10 | STOP_HIT | 1.00 | -4.38% |
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
