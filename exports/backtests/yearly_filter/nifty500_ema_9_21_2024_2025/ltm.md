# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 4360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 98 |
| ALERT2 | 97 |
| ALERT2_SKIP | 48 |
| ALERT3 | 228 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 117 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 98
- **Target hits / Stop hits / Partials:** 4 / 117 / 7
- **Avg / median % per leg:** -0.16% / -0.83%
- **Sum % (uncompounded):** -19.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 16 | 22.9% | 2 | 66 | 2 | -0.48% | -33.7% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 0 | 3 | 2 | 4.07% | 20.4% |
| BUY @ 3rd Alert (retest2) | 65 | 11 | 16.9% | 2 | 63 | 0 | -0.83% | -54.1% |
| SELL (all) | 58 | 14 | 24.1% | 2 | 51 | 5 | 0.24% | 13.9% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.38% | 1.4% |
| SELL @ 3rd Alert (retest2) | 57 | 13 | 22.8% | 2 | 50 | 5 | 0.22% | 12.5% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 4 | 2 | 3.62% | 21.7% |
| retest2 (combined) | 122 | 24 | 19.7% | 4 | 113 | 5 | -0.34% | -41.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 4651.00 | 4625.26 | 4624.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 4670.75 | 4644.92 | 4634.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 11:15:00 | 4753.20 | 4753.51 | 4735.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 11:45:00 | 4749.00 | 4753.51 | 4735.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 4727.00 | 4747.73 | 4735.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:45:00 | 4725.55 | 4747.73 | 4735.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 4729.95 | 4744.17 | 4735.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:30:00 | 4723.05 | 4744.17 | 4735.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 4734.50 | 4742.24 | 4735.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 4761.95 | 4742.24 | 4735.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 4748.30 | 4744.02 | 4737.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 11:15:00 | 4768.95 | 4744.02 | 4737.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 4809.75 | 4867.17 | 4867.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 4809.75 | 4867.17 | 4867.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 12:15:00 | 4795.10 | 4840.02 | 4853.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 4655.40 | 4638.68 | 4686.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 13:00:00 | 4655.40 | 4638.68 | 4686.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 4705.95 | 4649.17 | 4675.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 4705.95 | 4649.17 | 4675.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 4759.80 | 4671.30 | 4683.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 4759.80 | 4671.30 | 4683.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 4711.65 | 4693.92 | 4691.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 4733.00 | 4700.41 | 4695.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 4876.45 | 4917.74 | 4851.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 4876.45 | 4917.74 | 4851.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 4901.00 | 4911.51 | 4893.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 4968.50 | 4911.51 | 4893.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 5020.45 | 5035.36 | 5035.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 5020.45 | 5035.36 | 5035.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 10:15:00 | 5010.05 | 5023.86 | 5029.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 11:15:00 | 5052.15 | 5029.51 | 5031.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 5052.15 | 5029.51 | 5031.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 5052.15 | 5029.51 | 5031.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:30:00 | 5049.00 | 5029.51 | 5031.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 5053.95 | 5034.40 | 5033.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 5178.00 | 5070.89 | 5051.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 5111.75 | 5118.97 | 5092.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 5111.75 | 5118.97 | 5092.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 5111.75 | 5118.97 | 5092.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 5175.15 | 5112.39 | 5106.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:45:00 | 5164.00 | 5157.98 | 5138.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 13:15:00 | 5430.20 | 5448.68 | 5451.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 5430.20 | 5448.68 | 5451.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 15:15:00 | 5413.25 | 5438.22 | 5445.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 5402.00 | 5397.41 | 5416.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:15:00 | 5407.00 | 5397.41 | 5416.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 5409.90 | 5399.91 | 5416.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:15:00 | 5413.00 | 5399.91 | 5416.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 5408.00 | 5401.53 | 5415.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:30:00 | 5386.75 | 5394.12 | 5410.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 5396.10 | 5379.22 | 5391.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 13:00:00 | 5398.05 | 5382.99 | 5392.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:00:00 | 5383.80 | 5381.04 | 5388.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 5381.30 | 5381.09 | 5387.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 5361.85 | 5378.29 | 5385.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 13:00:00 | 5365.90 | 5375.81 | 5383.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 15:15:00 | 5410.00 | 5388.14 | 5387.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 5410.00 | 5388.14 | 5387.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 5536.00 | 5417.71 | 5401.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 10:15:00 | 5522.00 | 5539.66 | 5493.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 11:00:00 | 5522.00 | 5539.66 | 5493.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 5480.00 | 5525.23 | 5498.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 5480.00 | 5525.23 | 5498.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 5475.95 | 5515.38 | 5496.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:30:00 | 5473.00 | 5515.38 | 5496.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 5473.60 | 5501.68 | 5492.84 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 11:15:00 | 5463.40 | 5485.70 | 5486.60 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 5501.90 | 5488.94 | 5487.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 13:15:00 | 5529.95 | 5497.14 | 5491.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 15:15:00 | 5745.00 | 5758.36 | 5693.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:15:00 | 5778.35 | 5758.36 | 5693.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 5738.95 | 5754.48 | 5697.79 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 5646.05 | 5686.39 | 5689.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 5598.00 | 5659.89 | 5672.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 5752.00 | 5638.38 | 5648.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 5752.00 | 5638.38 | 5648.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 5752.00 | 5638.38 | 5648.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 5752.00 | 5638.38 | 5648.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 5757.60 | 5662.22 | 5658.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 5775.00 | 5700.11 | 5677.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 5699.10 | 5756.34 | 5735.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 5699.10 | 5756.34 | 5735.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 5699.10 | 5756.34 | 5735.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 5699.10 | 5756.34 | 5735.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 5721.25 | 5749.32 | 5734.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:45:00 | 5725.50 | 5745.71 | 5733.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 5694.80 | 5725.28 | 5726.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 5694.80 | 5725.28 | 5726.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 15:15:00 | 5667.00 | 5713.62 | 5721.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 5701.50 | 5684.45 | 5698.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 5701.50 | 5684.45 | 5698.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 5701.50 | 5684.45 | 5698.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 5728.60 | 5684.45 | 5698.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 5684.45 | 5684.45 | 5697.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:15:00 | 5669.80 | 5684.45 | 5697.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 5674.85 | 5671.24 | 5685.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 5386.31 | 5527.10 | 5597.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 5391.11 | 5527.10 | 5597.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 5486.00 | 5424.10 | 5494.01 | SL hit (close>ema200) qty=0.50 sl=5424.10 alert=retest2 |

### Cycle 13 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 5540.00 | 5500.26 | 5498.97 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 5491.00 | 5505.47 | 5505.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 5404.65 | 5485.31 | 5496.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 5435.10 | 5421.71 | 5457.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 5435.10 | 5421.71 | 5457.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 5435.10 | 5421.71 | 5457.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:45:00 | 5390.00 | 5414.48 | 5445.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 12:45:00 | 5396.00 | 5395.36 | 5417.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 15:00:00 | 5396.00 | 5400.91 | 5416.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 5375.00 | 5401.33 | 5414.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 5402.75 | 5390.39 | 5400.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-14 15:15:00 | 5432.05 | 5409.18 | 5406.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 5432.05 | 5409.18 | 5406.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 5568.10 | 5440.96 | 5420.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 5668.75 | 5694.39 | 5648.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:00:00 | 5668.75 | 5694.39 | 5648.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 5680.25 | 5712.45 | 5689.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:30:00 | 5673.10 | 5712.45 | 5689.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 5697.80 | 5709.52 | 5690.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 14:30:00 | 5713.75 | 5708.94 | 5691.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 5630.00 | 5691.74 | 5686.78 | SL hit (close<static) qty=1.00 sl=5670.05 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 5642.50 | 5681.89 | 5682.75 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 5730.00 | 5682.06 | 5678.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 5741.45 | 5708.10 | 5693.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 12:15:00 | 5723.00 | 5729.29 | 5711.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 13:00:00 | 5723.00 | 5729.29 | 5711.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 6127.20 | 6142.72 | 6108.98 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 6014.40 | 6101.53 | 6107.82 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 6131.00 | 6100.02 | 6098.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 13:15:00 | 6144.85 | 6108.98 | 6102.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 14:15:00 | 6164.20 | 6179.54 | 6151.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 15:00:00 | 6164.20 | 6179.54 | 6151.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 6171.65 | 6177.96 | 6153.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 6119.45 | 6177.96 | 6153.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 6159.95 | 6174.36 | 6154.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 6213.00 | 6152.62 | 6149.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 10:15:00 | 6299.00 | 6389.08 | 6398.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 6299.00 | 6389.08 | 6398.41 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 6437.00 | 6399.32 | 6395.28 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 6350.15 | 6388.41 | 6390.96 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 6442.40 | 6397.93 | 6393.86 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 6355.00 | 6385.94 | 6389.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 10:15:00 | 6301.90 | 6369.13 | 6381.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 14:15:00 | 6349.90 | 6293.36 | 6320.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 6349.90 | 6293.36 | 6320.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 6349.90 | 6293.36 | 6320.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 6349.90 | 6293.36 | 6320.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 6350.00 | 6304.69 | 6323.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 6313.50 | 6304.69 | 6323.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 6298.10 | 6170.19 | 6197.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 12:15:00 | 6216.20 | 6195.70 | 6193.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 12:15:00 | 6216.20 | 6195.70 | 6193.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 13:15:00 | 6267.15 | 6238.59 | 6220.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 6221.00 | 6244.05 | 6230.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 6221.00 | 6244.05 | 6230.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 6221.00 | 6244.05 | 6230.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 6221.00 | 6244.05 | 6230.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 6188.00 | 6232.84 | 6226.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 6190.00 | 6232.84 | 6226.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 6187.20 | 6216.50 | 6219.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 6115.25 | 6189.14 | 6203.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 6270.00 | 6182.74 | 6195.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 6270.00 | 6182.74 | 6195.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 6270.00 | 6182.74 | 6195.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:30:00 | 6292.00 | 6182.74 | 6195.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 6191.60 | 6184.51 | 6194.72 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 12:15:00 | 6260.15 | 6206.63 | 6203.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 10:15:00 | 6273.80 | 6242.90 | 6224.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 6422.30 | 6422.62 | 6364.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:00:00 | 6422.30 | 6422.62 | 6364.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 6346.00 | 6400.47 | 6372.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:30:00 | 6339.55 | 6400.47 | 6372.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 6347.05 | 6389.79 | 6369.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 6347.05 | 6389.79 | 6369.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 6399.45 | 6404.60 | 6385.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:00:00 | 6399.45 | 6404.60 | 6385.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 6368.00 | 6397.28 | 6383.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:45:00 | 6368.50 | 6397.28 | 6383.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 6410.55 | 6399.94 | 6386.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 6425.15 | 6396.71 | 6387.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:00:00 | 6425.80 | 6402.53 | 6390.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:00:00 | 6429.70 | 6455.89 | 6440.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:45:00 | 6415.80 | 6445.99 | 6437.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 6355.85 | 6427.97 | 6429.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 6355.85 | 6427.97 | 6429.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 6338.05 | 6373.77 | 6398.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 6388.55 | 6369.75 | 6391.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 13:00:00 | 6388.55 | 6369.75 | 6391.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 6358.75 | 6367.55 | 6388.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 6149.90 | 6380.91 | 6391.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 5842.40 | 5929.71 | 5957.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 5903.40 | 5902.61 | 5933.76 | SL hit (close>ema200) qty=0.50 sl=5902.61 alert=retest2 |

### Cycle 29 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 5867.00 | 5748.51 | 5743.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 5945.60 | 5787.92 | 5761.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 5878.55 | 5907.71 | 5845.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 5878.55 | 5907.71 | 5845.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 5832.00 | 5892.56 | 5844.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 5832.00 | 5892.56 | 5844.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 5830.00 | 5880.05 | 5842.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 5830.00 | 5880.05 | 5842.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 5859.90 | 5876.02 | 5844.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 13:45:00 | 5876.00 | 5876.74 | 5847.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 15:15:00 | 5937.70 | 5964.01 | 5964.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 5937.70 | 5964.01 | 5964.71 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 12:15:00 | 5990.45 | 5966.13 | 5964.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 14:15:00 | 5998.05 | 5976.07 | 5969.85 | Break + close above crossover candle high |

### Cycle 32 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 5797.00 | 5941.69 | 5955.39 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 14:15:00 | 5937.60 | 5907.10 | 5905.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 6057.00 | 5940.73 | 5921.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 6116.05 | 6119.80 | 6061.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 6116.05 | 6119.80 | 6061.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 6179.40 | 6236.08 | 6198.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 6187.65 | 6236.08 | 6198.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 6149.50 | 6218.77 | 6194.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 6149.50 | 6218.77 | 6194.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 6105.05 | 6196.02 | 6186.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 6095.65 | 6196.02 | 6186.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 6090.00 | 6174.82 | 6177.28 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 6215.90 | 6175.72 | 6175.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 6267.00 | 6210.73 | 6197.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 6174.10 | 6205.61 | 6197.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 11:15:00 | 6174.10 | 6205.61 | 6197.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 6174.10 | 6205.61 | 6197.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 6174.10 | 6205.61 | 6197.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 6129.10 | 6190.31 | 6191.30 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 11:15:00 | 6215.95 | 6187.56 | 6187.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 13:15:00 | 6229.00 | 6199.63 | 6193.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 15:15:00 | 6356.10 | 6360.88 | 6317.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 09:15:00 | 6412.45 | 6360.88 | 6317.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 6676.90 | 6716.66 | 6694.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:45:00 | 6666.85 | 6716.66 | 6694.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 6657.90 | 6704.91 | 6691.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 6657.90 | 6704.91 | 6691.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 6674.30 | 6698.79 | 6689.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 14:30:00 | 6706.50 | 6698.24 | 6690.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 6692.00 | 6693.18 | 6688.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 6650.50 | 6684.65 | 6685.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 6650.50 | 6684.65 | 6685.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 6633.70 | 6674.46 | 6680.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 5725.15 | 5720.91 | 5811.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 12:45:00 | 5735.20 | 5720.91 | 5811.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 5745.85 | 5737.11 | 5791.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 5727.10 | 5737.69 | 5786.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 15:15:00 | 5675.00 | 5651.91 | 5649.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 5675.00 | 5651.91 | 5649.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 5711.30 | 5663.79 | 5655.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 5709.10 | 5724.12 | 5699.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:45:00 | 5702.55 | 5724.12 | 5699.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 5733.45 | 5725.99 | 5702.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 12:30:00 | 5737.05 | 5731.29 | 5706.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 15:00:00 | 5739.20 | 5736.67 | 5713.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 5836.20 | 5736.25 | 5715.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:00:00 | 5755.05 | 5763.62 | 5735.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 5735.05 | 5759.36 | 5741.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 5731.10 | 5759.36 | 5741.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 5723.30 | 5752.15 | 5739.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 5828.35 | 5752.15 | 5739.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 5689.45 | 5751.08 | 5750.12 | SL hit (close<static) qty=1.00 sl=5700.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 5719.80 | 5744.82 | 5747.36 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 12:15:00 | 5785.00 | 5754.09 | 5751.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 5852.40 | 5773.75 | 5760.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 15:15:00 | 5829.00 | 5840.63 | 5814.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 15:15:00 | 5829.00 | 5840.63 | 5814.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 5829.00 | 5840.63 | 5814.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 6016.75 | 5840.63 | 5814.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 5828.20 | 5965.03 | 5976.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 5828.20 | 5965.03 | 5976.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 5735.10 | 5919.04 | 5954.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 11:15:00 | 5832.75 | 5819.20 | 5877.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 11:45:00 | 5829.45 | 5819.20 | 5877.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 5856.50 | 5826.66 | 5875.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:45:00 | 5879.00 | 5826.66 | 5875.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 5903.45 | 5846.02 | 5869.66 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 5968.55 | 5895.12 | 5887.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 5987.60 | 5925.56 | 5903.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 5806.50 | 5901.75 | 5894.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 5806.50 | 5901.75 | 5894.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 5806.50 | 5901.75 | 5894.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 5803.60 | 5901.75 | 5894.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 5848.05 | 5891.01 | 5890.21 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 5852.70 | 5883.35 | 5886.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 12:15:00 | 5812.00 | 5850.00 | 5866.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 5881.90 | 5846.76 | 5858.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 5881.90 | 5846.76 | 5858.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 5881.90 | 5846.76 | 5858.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 5881.90 | 5846.76 | 5858.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 5862.85 | 5849.98 | 5858.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:00:00 | 5816.55 | 5842.64 | 5853.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:30:00 | 5822.30 | 5808.63 | 5823.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 5994.50 | 5856.74 | 5842.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 5994.50 | 5856.74 | 5842.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 6037.00 | 5892.79 | 5860.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 5992.90 | 5999.25 | 5958.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 15:00:00 | 5992.90 | 5999.25 | 5958.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 5809.00 | 5957.32 | 5946.30 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 5812.10 | 5928.27 | 5934.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 5740.00 | 5869.46 | 5905.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 5834.95 | 5724.64 | 5770.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 5834.95 | 5724.64 | 5770.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 5834.95 | 5724.64 | 5770.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 5834.95 | 5724.64 | 5770.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 5849.90 | 5749.69 | 5777.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 5849.90 | 5749.69 | 5777.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 5905.25 | 5814.11 | 5802.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 15:15:00 | 5920.60 | 5845.62 | 5819.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 5834.15 | 5862.95 | 5840.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 5834.15 | 5862.95 | 5840.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 5834.15 | 5862.95 | 5840.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 5834.15 | 5862.95 | 5840.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 5834.60 | 5857.28 | 5840.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 5824.70 | 5857.28 | 5840.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 5861.80 | 5858.19 | 5842.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 5939.30 | 5858.19 | 5842.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 13:15:00 | 5871.65 | 5852.43 | 5843.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:15:00 | 5873.55 | 5855.63 | 5846.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 5891.65 | 5889.70 | 5871.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 5910.25 | 5893.81 | 5875.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 5916.65 | 5893.81 | 5875.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 5915.00 | 5895.05 | 5877.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 5765.60 | 5872.35 | 5870.26 | SL hit (close<static) qty=1.00 sl=5831.45 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 5783.45 | 5854.57 | 5862.37 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 5894.90 | 5857.79 | 5853.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 5942.75 | 5887.60 | 5869.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 5985.75 | 5993.13 | 5945.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 5985.75 | 5993.13 | 5945.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 5932.05 | 5974.39 | 5947.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 5932.05 | 5974.39 | 5947.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 5913.25 | 5962.16 | 5944.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 5913.25 | 5962.16 | 5944.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 5950.55 | 5957.73 | 5945.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 5904.90 | 5957.73 | 5945.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 5936.20 | 5953.43 | 5944.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 5895.85 | 5953.43 | 5944.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 5944.00 | 5951.54 | 5944.76 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 5892.05 | 5933.20 | 5937.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 5850.00 | 5915.72 | 5928.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 5706.05 | 5693.16 | 5744.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 5706.05 | 5693.16 | 5744.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 5678.05 | 5691.40 | 5734.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 5601.25 | 5671.91 | 5718.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 5601.95 | 5651.92 | 5700.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 5599.55 | 5651.92 | 5700.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:00:00 | 5572.50 | 5623.35 | 5674.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 5601.00 | 5506.05 | 5542.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:45:00 | 5600.45 | 5506.05 | 5542.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 5579.60 | 5520.76 | 5545.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 5579.60 | 5520.76 | 5545.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-18 13:15:00 | 5663.70 | 5578.01 | 5567.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 5663.70 | 5578.01 | 5567.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 5680.60 | 5598.53 | 5578.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 09:15:00 | 5554.95 | 5598.21 | 5582.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 5554.95 | 5598.21 | 5582.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 5554.95 | 5598.21 | 5582.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 5541.65 | 5598.21 | 5582.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 5532.15 | 5585.00 | 5577.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:00:00 | 5532.15 | 5585.00 | 5577.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 5501.80 | 5558.44 | 5566.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 5485.00 | 5543.75 | 5558.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 4776.30 | 4724.91 | 4809.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 4776.30 | 4724.91 | 4809.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 4805.00 | 4740.93 | 4809.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 4789.80 | 4740.93 | 4809.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 4831.80 | 4759.11 | 4811.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 4831.80 | 4759.11 | 4811.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 4831.00 | 4773.48 | 4813.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 4683.70 | 4773.48 | 4813.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 4831.25 | 4776.16 | 4769.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 4831.25 | 4776.16 | 4769.18 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 11:15:00 | 4722.55 | 4776.71 | 4779.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 15:15:00 | 4705.00 | 4746.91 | 4763.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 10:15:00 | 4741.60 | 4741.57 | 4758.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 4741.60 | 4741.57 | 4758.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 4451.60 | 4384.84 | 4436.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 4461.05 | 4384.84 | 4436.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 4436.00 | 4395.08 | 4436.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 4351.50 | 4429.31 | 4439.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:45:00 | 4416.00 | 4379.08 | 4400.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 13:15:00 | 4409.15 | 4394.89 | 4402.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 14:45:00 | 4416.15 | 4405.41 | 4406.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 4430.40 | 4410.85 | 4408.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 09:15:00 | 4430.40 | 4410.85 | 4408.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 11:15:00 | 4506.70 | 4432.76 | 4419.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 4576.40 | 4580.43 | 4533.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:30:00 | 4575.65 | 4580.43 | 4533.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 4646.00 | 4621.76 | 4595.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 4659.20 | 4631.03 | 4609.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 4536.00 | 4597.31 | 4600.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 4536.00 | 4597.31 | 4600.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 4496.30 | 4577.11 | 4591.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 13:15:00 | 4471.85 | 4434.93 | 4470.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 13:15:00 | 4471.85 | 4434.93 | 4470.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 4471.85 | 4434.93 | 4470.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 4471.85 | 4434.93 | 4470.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 4500.00 | 4447.95 | 4472.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 4500.00 | 4447.95 | 4472.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 4480.00 | 4454.36 | 4473.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 4426.00 | 4454.36 | 4473.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:15:00 | 4204.70 | 4286.66 | 4361.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 3983.40 | 4152.20 | 4261.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 4221.00 | 4146.57 | 4144.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 4234.10 | 4164.08 | 4152.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 4275.20 | 4289.50 | 4256.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 4275.20 | 4289.50 | 4256.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 4276.90 | 4286.98 | 4258.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 4264.20 | 4286.98 | 4258.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 09:15:00 | 4093.90 | 4245.23 | 4246.03 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 11:15:00 | 4339.50 | 4244.44 | 4234.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 09:15:00 | 4472.80 | 4343.28 | 4304.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 4472.80 | 4506.14 | 4461.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 4472.80 | 4506.14 | 4461.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 4472.80 | 4506.14 | 4461.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 4472.80 | 4506.14 | 4461.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 4507.90 | 4506.49 | 4466.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 4475.00 | 4506.49 | 4466.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 4528.70 | 4510.93 | 4471.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 13:30:00 | 4541.90 | 4518.44 | 4482.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 14:15:00 | 4552.80 | 4518.44 | 4482.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 10:15:00 | 4460.00 | 4493.48 | 4480.99 | SL hit (close<static) qty=1.00 sl=4464.20 alert=retest2 |

### Cycle 60 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 4538.80 | 4590.53 | 4592.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 4513.50 | 4548.25 | 4568.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 4568.30 | 4552.26 | 4568.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 4568.30 | 4552.26 | 4568.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 4568.30 | 4552.26 | 4568.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 4568.30 | 4552.26 | 4568.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 4575.30 | 4556.87 | 4568.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 4575.30 | 4556.87 | 4568.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 4575.70 | 4560.63 | 4569.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 4583.90 | 4560.63 | 4569.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 4584.50 | 4565.41 | 4570.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 4584.50 | 4565.41 | 4570.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 4561.40 | 4564.61 | 4570.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 15:15:00 | 4550.00 | 4564.61 | 4570.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 4617.50 | 4572.85 | 4572.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 4617.50 | 4572.85 | 4572.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 4626.30 | 4583.54 | 4577.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 4593.40 | 4602.54 | 4589.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 4593.40 | 4602.54 | 4589.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 4593.40 | 4602.54 | 4589.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 4593.40 | 4602.54 | 4589.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 4601.60 | 4602.35 | 4590.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 4601.60 | 4602.35 | 4590.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 4586.00 | 4599.08 | 4590.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 4590.00 | 4599.08 | 4590.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 4621.00 | 4603.47 | 4592.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 10:15:00 | 4633.90 | 4603.47 | 4592.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:00:00 | 4628.70 | 4608.51 | 4596.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 4719.40 | 4613.13 | 4603.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-20 09:15:00 | 5097.29 | 5038.05 | 5025.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 5020.80 | 5033.01 | 5034.34 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 5070.40 | 5040.48 | 5037.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 5087.90 | 5059.61 | 5051.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 5082.20 | 5098.51 | 5080.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 5082.20 | 5098.51 | 5080.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 5082.20 | 5098.51 | 5080.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 5082.20 | 5098.51 | 5080.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 5077.80 | 5094.37 | 5080.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 5097.30 | 5094.37 | 5080.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 5109.40 | 5088.59 | 5081.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 5089.20 | 5095.72 | 5087.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 5070.60 | 5109.60 | 5113.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 5070.60 | 5109.60 | 5113.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 5044.50 | 5076.68 | 5094.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 5080.50 | 5075.47 | 5089.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:30:00 | 5087.00 | 5075.47 | 5089.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 5085.00 | 5077.38 | 5088.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 5080.00 | 5077.38 | 5088.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 5075.00 | 5076.90 | 5087.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 5032.00 | 5067.82 | 5081.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 5136.00 | 5076.35 | 5076.83 | SL hit (close>static) qty=1.00 sl=5122.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 5143.50 | 5089.78 | 5082.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 5177.00 | 5140.59 | 5116.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 5170.00 | 5170.69 | 5147.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 5176.50 | 5170.69 | 5147.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5434.50 | 5419.88 | 5387.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 5449.00 | 5419.88 | 5387.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 5457.00 | 5431.42 | 5411.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 5440.00 | 5435.09 | 5416.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:30:00 | 5449.50 | 5479.62 | 5471.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 5273.00 | 5406.00 | 5438.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 5366.00 | 5349.91 | 5386.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 5366.00 | 5349.91 | 5386.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 5377.00 | 5350.39 | 5377.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 5377.00 | 5350.39 | 5377.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 5362.00 | 5352.71 | 5375.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:45:00 | 5340.00 | 5350.17 | 5372.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 5356.00 | 5362.21 | 5364.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 14:00:00 | 5356.50 | 5362.21 | 5364.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 14:45:00 | 5344.50 | 5358.17 | 5362.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 5330.00 | 5352.53 | 5359.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 5398.50 | 5352.53 | 5359.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 5402.50 | 5362.53 | 5363.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 5402.50 | 5362.53 | 5363.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 5459.00 | 5401.41 | 5382.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 5413.50 | 5417.65 | 5395.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 5414.00 | 5417.65 | 5395.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 5365.00 | 5407.12 | 5393.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 5365.00 | 5407.12 | 5393.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 5387.50 | 5403.19 | 5392.59 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 5375.00 | 5386.87 | 5387.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 10:15:00 | 5363.50 | 5380.62 | 5384.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 5311.00 | 5304.74 | 5332.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 5311.00 | 5304.74 | 5332.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 5314.00 | 5309.31 | 5327.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 5345.50 | 5309.31 | 5327.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 5317.00 | 5310.85 | 5326.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 5333.00 | 5310.85 | 5326.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 5320.00 | 5306.25 | 5317.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 5339.50 | 5306.25 | 5317.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 5321.50 | 5309.30 | 5317.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:15:00 | 5309.00 | 5309.30 | 5317.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 5307.00 | 5311.51 | 5317.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 5305.00 | 5309.30 | 5313.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 5314.00 | 5312.94 | 5314.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 5310.00 | 5319.39 | 5319.51 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 5360.00 | 5324.39 | 5321.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 5376.50 | 5347.91 | 5335.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 5344.50 | 5359.99 | 5346.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 13:15:00 | 5344.50 | 5359.99 | 5346.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 5344.50 | 5359.99 | 5346.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 5344.50 | 5359.99 | 5346.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5358.00 | 5359.59 | 5347.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 5377.00 | 5359.59 | 5347.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 5324.50 | 5355.36 | 5347.81 | SL hit (close<static) qty=1.00 sl=5336.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 5316.50 | 5338.96 | 5341.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 5233.00 | 5295.79 | 5313.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5200.00 | 5172.19 | 5207.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 5200.00 | 5172.19 | 5207.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5200.00 | 5172.19 | 5207.65 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 5248.00 | 5223.92 | 5223.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 5293.00 | 5242.07 | 5232.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 5238.00 | 5282.06 | 5262.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 5238.00 | 5282.06 | 5262.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 5238.00 | 5282.06 | 5262.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 5238.00 | 5282.06 | 5262.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5236.00 | 5272.84 | 5260.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 5210.00 | 5272.84 | 5260.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 5228.00 | 5249.30 | 5251.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 5203.00 | 5240.04 | 5247.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 5147.50 | 5140.95 | 5172.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 5170.00 | 5153.94 | 5166.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 5170.00 | 5153.94 | 5166.32 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 5178.00 | 5172.42 | 5171.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 5224.00 | 5183.07 | 5176.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 5224.00 | 5243.10 | 5215.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 5224.00 | 5243.10 | 5215.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 5231.00 | 5240.68 | 5217.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:30:00 | 5244.00 | 5232.61 | 5217.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 5201.00 | 5222.99 | 5215.19 | SL hit (close<static) qty=1.00 sl=5205.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 5126.50 | 5200.02 | 5205.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 5080.00 | 5176.01 | 5194.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 5073.00 | 5060.80 | 5092.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 5073.00 | 5060.80 | 5092.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 5083.50 | 5063.19 | 5085.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 5083.50 | 5063.19 | 5085.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5084.50 | 5067.45 | 5085.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 5101.00 | 5067.45 | 5085.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 5104.00 | 5074.76 | 5087.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 5113.50 | 5074.76 | 5087.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 5100.00 | 5079.81 | 5088.19 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 5138.00 | 5096.76 | 5094.77 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 5062.50 | 5100.93 | 5101.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 5020.00 | 5078.92 | 5090.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 5050.00 | 5025.78 | 5051.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 5050.00 | 5025.78 | 5051.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 5050.00 | 5025.78 | 5051.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 5050.00 | 5025.78 | 5051.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 5075.50 | 5035.73 | 5053.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 5075.50 | 5035.73 | 5053.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 5087.00 | 5045.98 | 5056.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 5089.50 | 5045.98 | 5056.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 5077.00 | 5059.90 | 5061.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 5068.50 | 5059.90 | 5061.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 5107.00 | 5069.32 | 5065.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 5144.00 | 5086.77 | 5074.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 5065.50 | 5091.29 | 5080.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 5065.50 | 5091.29 | 5080.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 5065.50 | 5091.29 | 5080.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 5063.00 | 5091.29 | 5080.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 5076.00 | 5088.23 | 5080.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 5066.50 | 5088.23 | 5080.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 5079.50 | 5086.48 | 5080.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 5076.50 | 5086.48 | 5080.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 5044.50 | 5078.09 | 5076.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 5044.50 | 5078.09 | 5076.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5034.00 | 5069.27 | 5073.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 5002.50 | 5052.24 | 5063.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 5037.50 | 5031.17 | 5049.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 5037.50 | 5031.17 | 5049.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 5044.50 | 5033.84 | 5048.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 5012.00 | 5033.84 | 5048.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 5030.50 | 5029.71 | 5042.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 5062.00 | 5031.18 | 5033.48 | SL hit (close>static) qty=1.00 sl=5051.50 alert=retest2 |

### Cycle 81 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 5064.50 | 5037.85 | 5036.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 5115.00 | 5055.78 | 5044.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 5092.50 | 5095.26 | 5074.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 5067.50 | 5087.26 | 5074.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 5067.50 | 5087.26 | 5074.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 5062.00 | 5087.26 | 5074.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 5106.00 | 5091.01 | 5077.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 5120.50 | 5099.09 | 5083.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 5178.50 | 5099.17 | 5084.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 5030.00 | 5099.28 | 5102.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 5030.00 | 5099.28 | 5102.46 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 5120.00 | 5098.38 | 5095.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 5133.50 | 5105.40 | 5099.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 5167.00 | 5171.30 | 5145.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 5167.00 | 5171.30 | 5145.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 5167.00 | 5171.30 | 5145.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 5234.00 | 5177.64 | 5150.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:00:00 | 5218.00 | 5185.71 | 5156.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:00:00 | 5225.00 | 5193.57 | 5162.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:00:00 | 5225.50 | 5216.28 | 5185.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 5277.00 | 5289.08 | 5259.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 5301.00 | 5289.86 | 5262.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 5300.00 | 5289.86 | 5262.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 5240.00 | 5278.39 | 5261.76 | SL hit (close<static) qty=1.00 sl=5250.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 5156.50 | 5240.51 | 5246.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 5153.00 | 5223.01 | 5238.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 5154.00 | 5153.04 | 5191.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:45:00 | 5184.00 | 5153.04 | 5191.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 5144.00 | 5135.16 | 5155.03 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 5200.00 | 5170.64 | 5167.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 5212.50 | 5179.01 | 5171.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 5209.50 | 5213.02 | 5193.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 5209.50 | 5213.02 | 5193.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 5209.00 | 5220.34 | 5202.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:15:00 | 5199.00 | 5220.34 | 5202.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 5155.50 | 5207.37 | 5198.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 5155.50 | 5207.37 | 5198.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 5170.00 | 5199.90 | 5195.59 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 5147.00 | 5189.32 | 5191.17 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 5196.00 | 5191.48 | 5191.16 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 5181.50 | 5189.48 | 5190.28 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 5218.50 | 5194.59 | 5192.31 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 5134.00 | 5185.36 | 5189.03 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 5215.00 | 5192.23 | 5190.92 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 5163.50 | 5186.48 | 5188.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 5140.00 | 5177.18 | 5184.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 5144.00 | 5137.42 | 5158.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 5144.00 | 5137.42 | 5158.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 5144.00 | 5137.42 | 5158.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 5166.50 | 5137.42 | 5158.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 5168.50 | 5141.73 | 5156.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 5168.50 | 5141.73 | 5156.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 5182.00 | 5149.78 | 5158.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 5182.00 | 5149.78 | 5158.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 5173.00 | 5165.77 | 5164.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 5350.00 | 5202.61 | 5181.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 5217.00 | 5278.48 | 5244.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 5217.00 | 5278.48 | 5244.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 5217.00 | 5278.48 | 5244.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 5217.00 | 5278.48 | 5244.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 5248.50 | 5272.48 | 5245.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:15:00 | 5272.50 | 5268.59 | 5245.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 5271.00 | 5434.87 | 5452.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 5271.00 | 5434.87 | 5452.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 5246.00 | 5310.16 | 5372.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 5293.50 | 5281.14 | 5327.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:15:00 | 5237.00 | 5281.14 | 5327.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 5164.50 | 5123.12 | 5148.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 5164.50 | 5123.12 | 5148.11 | SL hit (close>ema400) qty=1.00 sl=5148.11 alert=retest1 |

### Cycle 95 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 5145.50 | 5128.06 | 5126.24 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 14:15:00 | 5119.00 | 5124.64 | 5124.90 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 5225.00 | 5143.89 | 5133.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 5252.50 | 5165.61 | 5144.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 5330.00 | 5333.53 | 5294.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 12:00:00 | 5377.00 | 5346.62 | 5307.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 13:45:00 | 5385.50 | 5359.76 | 5320.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 5465.50 | 5491.07 | 5464.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 5465.50 | 5491.07 | 5464.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 5470.00 | 5486.86 | 5464.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5615.00 | 5486.86 | 5464.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:15:00 | 5645.85 | 5583.45 | 5535.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:15:00 | 5654.78 | 5598.06 | 5546.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 5615.00 | 5617.49 | 5577.43 | SL hit (close<ema200) qty=0.50 sl=5617.49 alert=retest1 |

### Cycle 98 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 5545.50 | 5596.57 | 5599.96 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 5635.00 | 5604.25 | 5603.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 5668.00 | 5623.44 | 5612.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 5608.50 | 5629.30 | 5618.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 5608.50 | 5629.30 | 5618.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 5608.50 | 5629.30 | 5618.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 5608.50 | 5629.30 | 5618.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 5602.00 | 5623.84 | 5617.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 5614.00 | 5623.84 | 5617.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 5631.50 | 5625.37 | 5618.53 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 5567.00 | 5610.76 | 5612.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 5525.00 | 5587.89 | 5601.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 5637.50 | 5584.98 | 5595.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 5637.50 | 5584.98 | 5595.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5637.50 | 5584.98 | 5595.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 5637.50 | 5584.98 | 5595.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 5617.00 | 5591.38 | 5597.74 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 5623.00 | 5604.68 | 5603.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 5644.00 | 5615.32 | 5608.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 5604.00 | 5619.67 | 5612.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 10:15:00 | 5604.00 | 5619.67 | 5612.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5604.00 | 5619.67 | 5612.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 5604.00 | 5619.67 | 5612.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5600.00 | 5615.73 | 5611.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:30:00 | 5605.50 | 5615.73 | 5611.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 5582.00 | 5607.59 | 5608.45 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 5634.00 | 5608.59 | 5608.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 5664.00 | 5625.18 | 5616.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 5645.50 | 5645.86 | 5630.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 5645.50 | 5645.86 | 5630.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 5645.50 | 5645.86 | 5630.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 5637.00 | 5645.86 | 5630.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 5656.50 | 5647.99 | 5632.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 5638.00 | 5647.99 | 5632.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 5680.00 | 5694.06 | 5677.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 5708.00 | 5694.06 | 5677.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 5711.00 | 5692.48 | 5679.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 5705.50 | 5692.45 | 5681.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 5614.00 | 5675.05 | 5678.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 5614.00 | 5675.05 | 5678.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 5594.00 | 5658.84 | 5670.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 5640.00 | 5633.30 | 5649.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 5640.00 | 5633.30 | 5649.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 5625.00 | 5631.64 | 5647.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 5625.00 | 5631.64 | 5647.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 5670.50 | 5639.41 | 5649.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 5670.50 | 5639.41 | 5649.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 5664.00 | 5644.33 | 5650.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:15:00 | 5641.00 | 5645.56 | 5650.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 5682.00 | 5607.27 | 5615.29 | SL hit (close>static) qty=1.00 sl=5673.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 5662.50 | 5628.91 | 5624.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 5712.50 | 5659.64 | 5643.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 5845.50 | 5866.78 | 5812.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 5845.50 | 5866.78 | 5812.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 5807.00 | 5852.14 | 5815.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 5810.00 | 5852.14 | 5815.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 5817.00 | 5845.11 | 5815.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 5830.50 | 5841.19 | 5816.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 5772.50 | 5827.45 | 5812.42 | SL hit (close<static) qty=1.00 sl=5798.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 5782.50 | 5815.75 | 5815.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 5756.00 | 5791.14 | 5803.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 5898.50 | 5808.03 | 5808.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 5898.50 | 5808.03 | 5808.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 5898.50 | 5808.03 | 5808.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 5898.50 | 5808.03 | 5808.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 5896.00 | 5825.62 | 5816.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 5983.00 | 5857.10 | 5831.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 5952.00 | 6010.86 | 5964.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 5952.00 | 6010.86 | 5964.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 5952.00 | 6010.86 | 5964.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 5954.00 | 6010.86 | 5964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 5936.50 | 5995.99 | 5961.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 5936.50 | 5995.99 | 5961.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 5926.50 | 5982.09 | 5958.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 5922.00 | 5982.09 | 5958.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 5940.50 | 5966.42 | 5959.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 5940.50 | 5966.42 | 5959.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 5939.50 | 5961.04 | 5957.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 5939.50 | 5961.04 | 5957.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 5918.00 | 5952.43 | 5953.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 5905.50 | 5943.04 | 5949.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 5860.00 | 5858.16 | 5891.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 5860.00 | 5858.16 | 5891.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 5860.00 | 5858.16 | 5891.35 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 5928.50 | 5896.04 | 5894.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 5947.00 | 5913.10 | 5902.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 6115.50 | 6120.04 | 6073.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:30:00 | 6160.50 | 6128.89 | 6094.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 6155.00 | 6157.51 | 6124.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 6137.00 | 6157.51 | 6124.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 6279.00 | 6297.36 | 6267.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 6279.00 | 6297.36 | 6267.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 6263.00 | 6290.49 | 6266.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 6263.00 | 6290.49 | 6266.79 | SL hit (close<ema400) qty=1.00 sl=6266.79 alert=retest1 |

### Cycle 110 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 6210.00 | 6250.46 | 6253.80 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 6266.00 | 6242.85 | 6242.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 6301.00 | 6254.48 | 6248.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 09:15:00 | 6268.00 | 6272.14 | 6260.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 6268.00 | 6272.14 | 6260.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 6268.00 | 6272.14 | 6260.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 6268.00 | 6272.14 | 6260.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 6274.00 | 6272.52 | 6261.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 6260.00 | 6272.52 | 6261.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 6289.50 | 6282.67 | 6270.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 6272.50 | 6282.67 | 6270.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 6247.50 | 6279.37 | 6271.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 6250.50 | 6279.37 | 6271.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 6291.00 | 6281.70 | 6272.96 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 6197.00 | 6264.42 | 6269.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 6108.00 | 6212.19 | 6228.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 6198.00 | 6189.30 | 6210.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 6198.00 | 6189.30 | 6210.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 6199.50 | 6191.34 | 6209.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 6185.50 | 6191.34 | 6209.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 6185.00 | 6190.07 | 6207.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 6235.00 | 6190.07 | 6207.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 6217.50 | 6195.56 | 6208.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 6188.50 | 6203.06 | 6209.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:45:00 | 6196.00 | 6202.59 | 6207.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 6142.00 | 6201.67 | 6207.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 6194.50 | 6192.77 | 6197.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 6173.00 | 6198.13 | 6199.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 6135.00 | 6181.08 | 6191.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 6060.50 | 6058.43 | 6088.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 6060.50 | 6058.43 | 6088.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 6060.50 | 6058.43 | 6088.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 6041.00 | 6058.12 | 6083.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:45:00 | 6044.00 | 6065.04 | 6076.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 6113.00 | 6079.61 | 6077.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 6113.00 | 6079.61 | 6077.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 6118.00 | 6087.29 | 6080.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 6087.50 | 6092.59 | 6085.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 6087.50 | 6092.59 | 6085.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 6087.50 | 6092.59 | 6085.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 6087.50 | 6092.59 | 6085.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 6075.00 | 6089.07 | 6084.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 6075.00 | 6089.07 | 6084.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 6072.00 | 6085.66 | 6083.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:15:00 | 6076.00 | 6085.66 | 6083.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 6068.50 | 6082.23 | 6081.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 6058.00 | 6082.23 | 6081.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 6071.50 | 6080.08 | 6080.98 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 6102.00 | 6080.86 | 6080.72 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 6053.00 | 6076.43 | 6078.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 6028.50 | 6064.51 | 6072.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 6082.50 | 6021.15 | 6039.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 6082.50 | 6021.15 | 6039.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 6082.50 | 6021.15 | 6039.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 6082.50 | 6021.15 | 6039.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 6095.00 | 6035.92 | 6044.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 6113.50 | 6035.92 | 6044.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 6116.00 | 6051.94 | 6051.37 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 6037.00 | 6056.82 | 6058.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 5951.00 | 6024.05 | 6038.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 6010.50 | 5994.39 | 6014.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 6010.50 | 5994.39 | 6014.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 6012.00 | 5997.92 | 6014.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 6038.50 | 5997.92 | 6014.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 6033.00 | 6004.93 | 6016.29 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 6084.50 | 6032.62 | 6027.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 6108.00 | 6061.07 | 6043.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 6033.00 | 6055.45 | 6042.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 6033.00 | 6055.45 | 6042.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 6033.00 | 6055.45 | 6042.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 6033.00 | 6055.45 | 6042.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 6024.00 | 6049.16 | 6040.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 6039.00 | 6049.16 | 6040.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 6049.00 | 6049.13 | 6041.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 6183.00 | 6039.84 | 6039.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 6007.50 | 6276.06 | 6238.64 | SL hit (close<static) qty=1.00 sl=6022.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 5921.50 | 6157.30 | 6188.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 5798.00 | 5999.42 | 6091.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5986.00 | 5894.07 | 5975.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5986.00 | 5894.07 | 5975.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5986.00 | 5894.07 | 5975.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 6010.00 | 5894.07 | 5975.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5942.50 | 5903.75 | 5972.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 5986.50 | 5903.75 | 5972.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 5950.00 | 5923.12 | 5956.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 6013.00 | 5923.12 | 5956.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 5955.00 | 5929.49 | 5956.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 5897.00 | 5921.18 | 5947.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 5985.00 | 5931.30 | 5927.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 5985.00 | 5931.30 | 5927.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 6010.00 | 5967.46 | 5948.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 5924.00 | 5966.53 | 5951.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 5924.00 | 5966.53 | 5951.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 5924.00 | 5966.53 | 5951.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 5913.00 | 5966.53 | 5951.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 5930.00 | 5959.23 | 5949.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:30:00 | 5966.00 | 5955.88 | 5948.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 5955.00 | 5959.75 | 5956.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:15:00 | 5955.00 | 5959.94 | 5958.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 5693.00 | 5971.36 | 5993.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 5693.00 | 5971.36 | 5993.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 5551.50 | 5663.95 | 5746.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 5605.00 | 5581.79 | 5645.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 5605.00 | 5581.79 | 5645.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 5635.00 | 5592.44 | 5644.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 5630.50 | 5592.44 | 5644.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 5627.00 | 5606.16 | 5642.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:30:00 | 5641.50 | 5606.16 | 5642.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 5648.00 | 5619.91 | 5640.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 5648.00 | 5619.91 | 5640.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 5656.50 | 5627.22 | 5641.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 5649.00 | 5627.22 | 5641.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 5659.00 | 5646.43 | 5647.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 5659.00 | 5646.43 | 5647.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 5615.50 | 5638.42 | 5643.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:00:00 | 5602.00 | 5630.59 | 5639.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 5321.90 | 5512.78 | 5576.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 5041.80 | 5231.08 | 5379.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 4338.90 | 4302.34 | 4301.71 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 4224.80 | 4291.33 | 4297.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 4184.60 | 4227.29 | 4256.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 4215.30 | 4206.61 | 4235.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 4215.70 | 4206.61 | 4235.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 4179.00 | 4197.97 | 4226.51 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 4330.90 | 4226.60 | 4226.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 4347.10 | 4250.70 | 4237.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4267.40 | 4288.02 | 4267.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4267.40 | 4288.02 | 4267.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4267.40 | 4288.02 | 4267.47 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 4207.00 | 4253.54 | 4256.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 4188.00 | 4240.43 | 4250.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 4199.50 | 4190.23 | 4214.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 13:30:00 | 4195.00 | 4190.23 | 4214.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 4258.80 | 4203.94 | 4218.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 4258.80 | 4203.94 | 4218.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 4355.00 | 4234.15 | 4231.12 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 4131.70 | 4213.66 | 4222.08 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 4209.40 | 4180.11 | 4179.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4249.00 | 4200.59 | 4189.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 4265.70 | 4267.39 | 4239.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:45:00 | 4260.30 | 4267.39 | 4239.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 4250.00 | 4266.11 | 4243.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 4250.00 | 4266.11 | 4243.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 4212.10 | 4255.30 | 4240.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 4212.10 | 4255.30 | 4240.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 4206.50 | 4245.54 | 4237.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 4206.50 | 4245.54 | 4237.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4130.60 | 4213.27 | 4223.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4089.70 | 4188.55 | 4211.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4228.20 | 4121.69 | 4159.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4228.20 | 4121.69 | 4159.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4228.20 | 4121.69 | 4159.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 4228.20 | 4121.69 | 4159.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4197.30 | 4136.81 | 4162.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 4226.60 | 4136.81 | 4162.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 4184.80 | 4129.97 | 4143.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:45:00 | 4184.60 | 4129.97 | 4143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 4240.50 | 4152.08 | 4152.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 4240.50 | 4152.08 | 4152.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 4300.30 | 4181.72 | 4166.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 4314.20 | 4208.22 | 4179.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 4432.30 | 4477.34 | 4417.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 4432.30 | 4477.34 | 4417.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4467.10 | 4526.12 | 4477.92 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 4461.80 | 4472.63 | 4473.00 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 4578.00 | 4491.89 | 4481.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 4657.00 | 4577.81 | 4532.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 4700.00 | 4727.81 | 4681.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 4713.00 | 4727.81 | 4681.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 4713.00 | 4724.85 | 4684.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 4761.20 | 4729.84 | 4708.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 14:45:00 | 4760.60 | 4739.12 | 4718.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 4646.20 | 4723.46 | 4715.01 | SL hit (close<static) qty=1.00 sl=4680.70 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 4603.00 | 4699.37 | 4704.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 4575.00 | 4674.49 | 4693.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 4342.00 | 4341.75 | 4422.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 4342.00 | 4341.75 | 4422.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4370.40 | 4349.59 | 4395.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 4334.90 | 4356.59 | 4374.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 4349.40 | 4285.61 | 4278.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 4349.40 | 4285.61 | 4278.97 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 4251.50 | 4280.41 | 4283.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 12:15:00 | 4244.70 | 4273.27 | 4279.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 4302.10 | 4267.93 | 4273.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 4302.10 | 4267.93 | 4273.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4302.10 | 4267.93 | 4273.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:15:00 | 4346.30 | 4267.93 | 4273.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 4324.80 | 4279.31 | 4278.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 15:15:00 | 4360.00 | 4326.41 | 4304.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 11:15:00 | 4768.95 | 2024-05-30 09:15:00 | 4809.75 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-06-12 09:15:00 | 4968.50 | 2024-06-19 13:15:00 | 5020.45 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2024-06-26 09:15:00 | 5175.15 | 2024-07-05 13:15:00 | 5430.20 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2024-06-27 09:45:00 | 5164.00 | 2024-07-05 13:15:00 | 5430.20 | STOP_HIT | 1.00 | 5.15% |
| SELL | retest2 | 2024-07-09 11:30:00 | 5386.75 | 2024-07-11 15:15:00 | 5410.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-07-10 12:15:00 | 5396.10 | 2024-07-11 15:15:00 | 5410.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-07-10 13:00:00 | 5398.05 | 2024-07-11 15:15:00 | 5410.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-11 10:00:00 | 5383.80 | 2024-07-11 15:15:00 | 5410.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-07-11 11:45:00 | 5361.85 | 2024-07-11 15:15:00 | 5410.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-07-11 13:00:00 | 5365.90 | 2024-07-11 15:15:00 | 5410.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-30 11:45:00 | 5725.50 | 2024-07-30 14:15:00 | 5694.80 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-08-01 11:15:00 | 5669.80 | 2024-08-05 09:15:00 | 5386.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 15:15:00 | 5674.85 | 2024-08-05 09:15:00 | 5391.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 11:15:00 | 5669.80 | 2024-08-06 09:15:00 | 5486.00 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2024-08-01 15:15:00 | 5674.85 | 2024-08-06 09:15:00 | 5486.00 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2024-08-09 12:45:00 | 5390.00 | 2024-08-14 15:15:00 | 5432.05 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-08-12 12:45:00 | 5396.00 | 2024-08-14 15:15:00 | 5432.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-08-12 15:00:00 | 5396.00 | 2024-08-14 15:15:00 | 5432.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-08-13 09:15:00 | 5375.00 | 2024-08-14 15:15:00 | 5432.05 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-08-22 14:30:00 | 5713.75 | 2024-08-23 09:15:00 | 5630.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-09-10 09:15:00 | 6213.00 | 2024-09-18 10:15:00 | 6299.00 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2024-09-25 09:15:00 | 6313.50 | 2024-09-30 12:15:00 | 6216.20 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2024-09-27 09:30:00 | 6298.10 | 2024-09-30 12:15:00 | 6216.20 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-10-14 09:30:00 | 6425.15 | 2024-10-16 11:15:00 | 6355.85 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-10-14 11:00:00 | 6425.80 | 2024-10-16 11:15:00 | 6355.85 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-10-16 10:00:00 | 6429.70 | 2024-10-16 11:15:00 | 6355.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-16 10:45:00 | 6415.80 | 2024-10-16 11:15:00 | 6355.85 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-10-18 09:15:00 | 6149.90 | 2024-10-25 10:15:00 | 5842.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 09:15:00 | 6149.90 | 2024-10-25 14:15:00 | 5903.40 | STOP_HIT | 0.50 | 4.01% |
| BUY | retest2 | 2024-11-07 13:45:00 | 5876.00 | 2024-11-13 15:15:00 | 5937.70 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2024-12-17 14:30:00 | 6706.50 | 2024-12-18 09:15:00 | 6650.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-12-18 09:15:00 | 6692.00 | 2024-12-18 09:15:00 | 6650.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-27 11:15:00 | 5727.10 | 2025-01-01 15:15:00 | 5675.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-01-03 12:30:00 | 5737.05 | 2025-01-08 09:15:00 | 5689.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-01-03 15:00:00 | 5739.20 | 2025-01-08 09:15:00 | 5689.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-01-06 09:15:00 | 5836.20 | 2025-01-08 09:15:00 | 5689.45 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-01-06 12:00:00 | 5755.05 | 2025-01-08 09:15:00 | 5689.45 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-01-07 09:15:00 | 5828.35 | 2025-01-08 09:15:00 | 5689.45 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-01-10 09:15:00 | 6016.75 | 2025-01-14 11:15:00 | 5828.20 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-01-21 14:00:00 | 5816.55 | 2025-01-23 09:15:00 | 5994.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-01-22 13:30:00 | 5822.30 | 2025-01-23 09:15:00 | 5994.50 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-01-31 09:15:00 | 5939.30 | 2025-02-03 09:15:00 | 5765.60 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-01-31 13:15:00 | 5871.65 | 2025-02-03 09:15:00 | 5765.60 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-01-31 14:15:00 | 5873.55 | 2025-02-03 09:15:00 | 5765.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-02-01 13:15:00 | 5891.65 | 2025-02-03 09:15:00 | 5765.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-02-01 14:15:00 | 5916.65 | 2025-02-03 09:15:00 | 5765.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-02-01 15:15:00 | 5915.00 | 2025-02-03 09:15:00 | 5765.60 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-13 12:00:00 | 5601.25 | 2025-02-18 13:15:00 | 5663.70 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-02-13 13:30:00 | 5601.95 | 2025-02-18 13:15:00 | 5663.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-02-13 14:15:00 | 5599.55 | 2025-02-18 13:15:00 | 5663.70 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-02-14 10:00:00 | 5572.50 | 2025-02-18 13:15:00 | 5663.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-03-04 09:15:00 | 4683.70 | 2025-03-06 09:15:00 | 4831.25 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-03-19 09:15:00 | 4351.50 | 2025-03-21 09:15:00 | 4430.40 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-03-20 09:45:00 | 4416.00 | 2025-03-21 09:15:00 | 4430.40 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-03-20 13:15:00 | 4409.15 | 2025-03-21 09:15:00 | 4430.40 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-03-20 14:45:00 | 4416.15 | 2025-03-21 09:15:00 | 4430.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-03-27 15:00:00 | 4659.20 | 2025-03-28 12:15:00 | 4536.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-04-03 09:15:00 | 4426.00 | 2025-04-04 11:15:00 | 4204.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 4426.00 | 2025-04-07 09:15:00 | 3983.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-25 13:30:00 | 4541.90 | 2025-04-28 10:15:00 | 4460.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-04-25 14:15:00 | 4552.80 | 2025-04-28 10:15:00 | 4460.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-04-29 10:30:00 | 4539.10 | 2025-05-06 11:15:00 | 4538.80 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-05-07 15:15:00 | 4550.00 | 2025-05-08 09:15:00 | 4617.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-05-09 10:15:00 | 4633.90 | 2025-05-20 09:15:00 | 5097.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 11:00:00 | 4628.70 | 2025-05-20 09:15:00 | 5091.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 4719.40 | 2025-05-22 15:15:00 | 5020.80 | STOP_HIT | 1.00 | 6.39% |
| BUY | retest2 | 2025-05-27 11:15:00 | 5097.30 | 2025-05-30 11:15:00 | 5070.60 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-05-27 14:45:00 | 5109.40 | 2025-05-30 11:15:00 | 5070.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-28 11:30:00 | 5089.20 | 2025-05-30 11:15:00 | 5070.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-06-03 09:30:00 | 5032.00 | 2025-06-04 09:15:00 | 5136.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-06-13 10:15:00 | 5449.00 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-06-16 10:15:00 | 5457.00 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-06-16 12:15:00 | 5440.00 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-06-18 13:30:00 | 5449.50 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-23 09:45:00 | 5340.00 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-06-24 13:30:00 | 5356.00 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-24 14:00:00 | 5356.50 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-24 14:45:00 | 5344.50 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-07-02 10:15:00 | 5309.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-02 12:15:00 | 5307.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-03 09:15:00 | 5305.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-03 09:45:00 | 5314.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-08 15:15:00 | 5377.00 | 2025-07-09 09:15:00 | 5324.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-24 12:30:00 | 5244.00 | 2025-07-24 14:15:00 | 5201.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-08 09:15:00 | 5012.00 | 2025-08-11 13:15:00 | 5062.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-08-08 12:15:00 | 5030.50 | 2025-08-11 13:15:00 | 5062.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-13 14:45:00 | 5120.50 | 2025-08-18 11:15:00 | 5030.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-14 09:15:00 | 5178.50 | 2025-08-18 11:15:00 | 5030.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-08-21 10:30:00 | 5234.00 | 2025-08-26 12:15:00 | 5240.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-08-21 12:00:00 | 5218.00 | 2025-08-26 12:15:00 | 5240.00 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-08-21 13:00:00 | 5225.00 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-22 10:00:00 | 5225.50 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-26 10:30:00 | 5301.00 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-08-26 11:15:00 | 5300.00 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-09-11 12:15:00 | 5272.50 | 2025-09-22 10:15:00 | 5271.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest1 | 2025-09-24 09:15:00 | 5237.00 | 2025-09-29 14:15:00 | 5164.50 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-09-30 09:15:00 | 5122.50 | 2025-10-03 12:15:00 | 5145.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-10-01 09:15:00 | 5085.50 | 2025-10-03 12:15:00 | 5145.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2025-10-09 12:00:00 | 5377.00 | 2025-10-16 09:15:00 | 5645.85 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-09 13:45:00 | 5385.50 | 2025-10-16 10:15:00 | 5654.78 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-09 12:00:00 | 5377.00 | 2025-10-16 15:15:00 | 5615.00 | STOP_HIT | 0.50 | 4.43% |
| BUY | retest1 | 2025-10-09 13:45:00 | 5385.50 | 2025-10-16 15:15:00 | 5615.00 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2025-10-15 09:15:00 | 5615.00 | 2025-10-21 14:15:00 | 5545.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-03 09:15:00 | 5708.00 | 2025-11-04 10:15:00 | 5614.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-03 11:15:00 | 5711.00 | 2025-11-04 10:15:00 | 5614.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-03 13:00:00 | 5705.50 | 2025-11-04 10:15:00 | 5614.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-06 15:15:00 | 5641.00 | 2025-11-10 10:15:00 | 5682.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-14 11:45:00 | 5830.50 | 2025-11-14 12:15:00 | 5772.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-17 09:15:00 | 5836.00 | 2025-11-18 09:15:00 | 5788.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-17 11:45:00 | 5828.50 | 2025-11-18 09:15:00 | 5788.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-11-17 13:30:00 | 5835.00 | 2025-11-18 09:15:00 | 5788.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2025-12-02 14:30:00 | 6160.50 | 2025-12-08 13:15:00 | 6263.00 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-12-22 12:15:00 | 6188.50 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-22 14:45:00 | 6196.00 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-12-23 09:15:00 | 6142.00 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-24 09:15:00 | 6194.50 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-30 12:30:00 | 6041.00 | 2026-01-01 14:15:00 | 6113.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-31 11:45:00 | 6044.00 | 2026-01-01 14:15:00 | 6113.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-01-16 09:15:00 | 6183.00 | 2026-01-20 09:15:00 | 6007.50 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-01-23 11:30:00 | 5897.00 | 2026-01-28 09:15:00 | 5985.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-01-29 11:30:00 | 5966.00 | 2026-02-04 09:15:00 | 5693.00 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2026-01-30 10:30:00 | 5955.00 | 2026-02-04 09:15:00 | 5693.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2026-02-01 10:15:00 | 5955.00 | 2026-02-04 09:15:00 | 5693.00 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2026-02-11 12:00:00 | 5602.00 | 2026-02-12 09:15:00 | 5321.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 12:00:00 | 5602.00 | 2026-02-13 09:15:00 | 5041.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 11:45:00 | 4761.20 | 2026-04-22 09:15:00 | 4646.20 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-04-21 14:45:00 | 4760.60 | 2026-04-22 09:15:00 | 4646.20 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-04-29 13:30:00 | 4334.90 | 2026-05-06 09:15:00 | 4349.40 | STOP_HIT | 1.00 | -0.33% |
