# Pfizer Ltd. (PFIZER)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4793.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 49 |
| ALERT2 | 47 |
| ALERT2_SKIP | 29 |
| ALERT3 | 128 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 50 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 42
- **Target hits / Stop hits / Partials:** 0 / 52 / 2
- **Avg / median % per leg:** -0.53% / -1.07%
- **Sum % (uncompounded):** -28.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 2 | 8.7% | 0 | 23 | 0 | -1.07% | -24.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.46% | -0.5% |
| BUY @ 3rd Alert (retest2) | 22 | 2 | 9.1% | 0 | 22 | 0 | -1.10% | -24.2% |
| SELL (all) | 31 | 10 | 32.3% | 0 | 29 | 2 | -0.13% | -4.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| SELL @ 3rd Alert (retest2) | 30 | 10 | 33.3% | 0 | 28 | 2 | -0.10% | -3.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.76% | -1.5% |
| retest2 (combined) | 52 | 12 | 23.1% | 0 | 50 | 2 | -0.52% | -27.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 4258.50 | 4252.41 | 4252.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 4290.10 | 4261.03 | 4256.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 4430.10 | 4436.36 | 4393.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:30:00 | 4431.60 | 4436.36 | 4393.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 4420.00 | 4428.28 | 4405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 4411.00 | 4428.28 | 4405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 4462.10 | 4432.12 | 4411.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 4427.80 | 4432.12 | 4411.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 5742.00 | 5776.04 | 5744.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 5742.00 | 5776.04 | 5744.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 5730.50 | 5766.93 | 5743.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 5730.50 | 5766.93 | 5743.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 5708.00 | 5755.14 | 5740.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 5708.00 | 5755.14 | 5740.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 5694.50 | 5743.02 | 5736.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 5694.50 | 5743.02 | 5736.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 5704.50 | 5728.19 | 5730.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 09:15:00 | 5660.00 | 5714.55 | 5724.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 5741.00 | 5671.82 | 5690.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 5741.00 | 5671.82 | 5690.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 5741.00 | 5671.82 | 5690.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 5741.00 | 5671.82 | 5690.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 5764.00 | 5690.25 | 5697.18 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 5773.50 | 5706.90 | 5704.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 5780.00 | 5750.54 | 5730.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 5753.00 | 5766.01 | 5747.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:15:00 | 5870.00 | 5766.01 | 5747.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5874.00 | 5856.65 | 5816.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 5900.50 | 5856.65 | 5816.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:00:00 | 5887.50 | 5870.99 | 5836.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 5889.00 | 5870.79 | 5839.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 5889.00 | 5870.79 | 5839.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5843.00 | 5868.14 | 5844.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 5843.00 | 5868.14 | 5844.18 | SL hit (close<ema400) qty=1.00 sl=5844.18 alert=retest1 |

### Cycle 4 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 5803.00 | 5848.49 | 5854.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 5774.50 | 5817.02 | 5836.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 5779.00 | 5726.38 | 5768.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 5779.00 | 5726.38 | 5768.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 5779.00 | 5726.38 | 5768.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 5760.00 | 5726.38 | 5768.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 5742.50 | 5729.60 | 5766.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 5694.50 | 5724.28 | 5760.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 5650.00 | 5613.65 | 5610.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 5650.00 | 5613.65 | 5610.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 5670.50 | 5625.02 | 5616.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 5647.50 | 5660.41 | 5641.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 5647.50 | 5660.41 | 5641.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 5647.50 | 5660.41 | 5641.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:45:00 | 5640.00 | 5660.41 | 5641.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 5656.50 | 5659.63 | 5643.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 5646.00 | 5659.63 | 5643.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 5704.00 | 5697.34 | 5686.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 5777.50 | 5725.70 | 5708.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 5669.50 | 5807.17 | 5801.68 | SL hit (close<static) qty=1.00 sl=5685.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 5650.00 | 5775.74 | 5787.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 5649.00 | 5750.39 | 5775.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 5521.00 | 5518.34 | 5576.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:45:00 | 5529.50 | 5518.34 | 5576.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 5335.50 | 5363.74 | 5391.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 5318.50 | 5352.23 | 5381.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 5325.50 | 5336.14 | 5356.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 5331.00 | 5340.99 | 5354.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 5330.00 | 5339.40 | 5352.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 5267.50 | 5323.51 | 5342.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 5407.50 | 5309.97 | 5306.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 09:15:00 | 5407.50 | 5309.97 | 5306.68 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 5305.00 | 5330.79 | 5331.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 15:15:00 | 5265.00 | 5317.63 | 5325.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 5281.50 | 5276.54 | 5297.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 5281.50 | 5276.54 | 5297.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 5281.50 | 5276.54 | 5297.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 5281.50 | 5276.54 | 5297.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5292.50 | 5278.66 | 5292.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 5292.50 | 5278.66 | 5292.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 5289.00 | 5280.73 | 5292.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 5285.00 | 5280.73 | 5292.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 5124.00 | 5198.81 | 5233.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 5096.50 | 5166.48 | 5211.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 5098.00 | 5024.04 | 5016.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 5098.00 | 5024.04 | 5016.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 5430.50 | 5165.20 | 5114.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 5650.00 | 5668.39 | 5567.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 5660.00 | 5668.39 | 5567.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 5688.50 | 5688.79 | 5652.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 5730.00 | 5699.43 | 5660.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 5641.50 | 5685.83 | 5671.62 | SL hit (close<static) qty=1.00 sl=5644.50 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 5565.00 | 5661.66 | 5661.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 5522.50 | 5617.16 | 5640.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 5144.50 | 5144.35 | 5185.72 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 11:15:00 | 5132.50 | 5144.35 | 5185.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 5149.50 | 5127.94 | 5157.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 5143.00 | 5127.94 | 5157.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 5187.00 | 5141.36 | 5158.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 5187.00 | 5141.36 | 5158.48 | SL hit (close>ema400) qty=1.00 sl=5158.48 alert=retest1 |

### Cycle 11 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 5214.50 | 5171.67 | 5168.23 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 5141.00 | 5166.47 | 5168.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 5097.50 | 5140.08 | 5155.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 5153.00 | 5134.52 | 5146.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 5153.00 | 5134.52 | 5146.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 5153.00 | 5134.52 | 5146.69 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 5173.50 | 5148.92 | 5146.51 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 5143.00 | 5145.46 | 5145.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 5126.00 | 5138.59 | 5142.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 5139.50 | 5136.33 | 5140.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 5139.50 | 5136.33 | 5140.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 5162.50 | 5141.56 | 5142.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 5163.00 | 5141.56 | 5142.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 5167.00 | 5146.65 | 5144.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 5210.50 | 5159.42 | 5150.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 5197.00 | 5201.16 | 5180.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:30:00 | 5182.00 | 5201.16 | 5180.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 5190.00 | 5198.93 | 5181.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 5186.00 | 5198.93 | 5181.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 5189.50 | 5197.04 | 5182.36 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 10:15:00 | 5164.50 | 5174.40 | 5175.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 11:15:00 | 5149.50 | 5169.42 | 5173.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 5174.00 | 5157.58 | 5164.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 5174.00 | 5157.58 | 5164.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 5174.00 | 5157.58 | 5164.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 5171.00 | 5157.58 | 5164.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 5184.00 | 5162.87 | 5166.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 5184.00 | 5162.87 | 5166.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 5196.50 | 5169.59 | 5168.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 5225.00 | 5191.79 | 5180.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 15:15:00 | 5193.00 | 5198.65 | 5189.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 15:15:00 | 5193.00 | 5198.65 | 5189.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 5193.00 | 5198.65 | 5189.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 5121.50 | 5188.72 | 5186.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 5146.50 | 5180.27 | 5182.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 5138.00 | 5165.54 | 5175.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 5139.50 | 5138.68 | 5154.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 5139.50 | 5138.68 | 5154.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4958.00 | 4954.18 | 4980.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 4980.50 | 4954.18 | 4980.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 4979.00 | 4960.63 | 4978.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 4979.00 | 4960.63 | 4978.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 4977.00 | 4963.91 | 4978.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 4977.00 | 4963.91 | 4978.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 4990.50 | 4969.23 | 4979.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 4990.50 | 4969.23 | 4979.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 4878.50 | 4951.08 | 4970.52 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 5014.50 | 4975.81 | 4975.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 5032.50 | 4993.34 | 4983.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 5220.00 | 5223.46 | 5188.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 5220.00 | 5223.46 | 5188.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5220.00 | 5223.46 | 5188.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 5240.00 | 5223.46 | 5188.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 5236.00 | 5226.80 | 5201.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 5237.50 | 5225.54 | 5202.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 5244.50 | 5229.32 | 5208.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 5185.00 | 5222.76 | 5213.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 5185.00 | 5222.76 | 5213.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 5161.00 | 5210.41 | 5209.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 5161.00 | 5210.41 | 5209.13 | SL hit (close<static) qty=1.00 sl=5175.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 5329.50 | 5379.60 | 5382.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 5320.50 | 5367.78 | 5376.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 13:15:00 | 5372.50 | 5326.62 | 5341.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 5372.50 | 5326.62 | 5341.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 5372.50 | 5326.62 | 5341.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 5372.50 | 5326.62 | 5341.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 5359.00 | 5333.10 | 5343.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 5325.00 | 5332.78 | 5341.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:00:00 | 5328.00 | 5331.82 | 5340.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 5322.50 | 5330.27 | 5335.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:30:00 | 5310.00 | 5328.23 | 5333.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5330.50 | 5328.69 | 5333.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 5426.00 | 5348.15 | 5341.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 5426.00 | 5348.15 | 5341.87 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 5284.00 | 5331.72 | 5337.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 5252.00 | 5297.07 | 5310.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 12:15:00 | 5279.50 | 5277.71 | 5296.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 5279.50 | 5277.71 | 5296.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 5297.50 | 5280.84 | 5294.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:00:00 | 5270.00 | 5278.22 | 5290.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 5269.50 | 5274.49 | 5285.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 5264.50 | 5272.49 | 5283.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 5325.50 | 5259.75 | 5268.32 | SL hit (close>static) qty=1.00 sl=5324.50 alert=retest2 |

### Cycle 23 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5118.50 | 5101.17 | 5100.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 5138.50 | 5108.64 | 5104.39 | Break + close above crossover candle high |

### Cycle 24 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 5064.50 | 5100.27 | 5101.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 5024.50 | 5085.11 | 5094.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 5103.50 | 5041.18 | 5061.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 5103.50 | 5041.18 | 5061.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 5103.50 | 5041.18 | 5061.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 5103.50 | 5041.18 | 5061.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 4999.00 | 5032.74 | 5055.88 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 5100.00 | 5053.08 | 5051.78 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 5048.00 | 5062.23 | 5063.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 5010.50 | 5050.38 | 5057.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 5035.00 | 5032.50 | 5045.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 5035.00 | 5032.50 | 5045.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 5049.00 | 5035.80 | 5045.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 5049.00 | 5035.80 | 5045.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 5028.00 | 5034.24 | 5043.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:15:00 | 5020.50 | 5034.24 | 5043.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:00:00 | 5025.00 | 5018.82 | 5027.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 5029.00 | 4998.60 | 4996.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 5029.00 | 4998.60 | 4996.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 5050.50 | 5008.98 | 5001.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 13:15:00 | 5005.00 | 5010.75 | 5003.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 13:15:00 | 5005.00 | 5010.75 | 5003.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 5005.00 | 5010.75 | 5003.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 5005.00 | 5010.75 | 5003.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 4997.50 | 5008.10 | 5003.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:45:00 | 4995.00 | 5008.10 | 5003.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 4973.00 | 5001.08 | 5000.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5048.00 | 5001.08 | 5000.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 4982.00 | 4997.26 | 4998.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 4982.00 | 4997.26 | 4998.82 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 5035.00 | 5003.51 | 5000.90 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 4995.00 | 4998.69 | 4998.97 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 5022.50 | 5003.45 | 5001.11 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 4990.00 | 4998.89 | 4999.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 4975.50 | 4993.03 | 4996.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 4987.70 | 4973.61 | 4981.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 4987.70 | 4973.61 | 4981.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 4987.70 | 4973.61 | 4981.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 5023.60 | 4973.61 | 4981.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 4995.10 | 4977.91 | 4982.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 4995.10 | 4977.91 | 4982.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 4967.40 | 4975.81 | 4981.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 4963.00 | 4973.25 | 4979.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 4958.00 | 4976.75 | 4979.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 4965.70 | 4960.92 | 4967.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 4999.60 | 4973.28 | 4972.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 4999.60 | 4973.28 | 4972.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 15:15:00 | 5005.00 | 4987.32 | 4980.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 4995.00 | 4996.14 | 4988.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 4995.00 | 4996.14 | 4988.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 4995.00 | 4996.14 | 4988.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 4951.00 | 4996.14 | 4988.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 4957.00 | 4988.31 | 4985.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 4974.70 | 4988.31 | 4985.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 4938.90 | 4978.43 | 4981.71 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 4999.00 | 4980.26 | 4978.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 5001.80 | 4987.76 | 4983.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 5017.00 | 5019.41 | 5004.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 5011.10 | 5019.41 | 5004.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 5018.20 | 5019.17 | 5005.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 5005.90 | 5019.17 | 5005.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 5011.50 | 5017.63 | 5006.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 5011.50 | 5017.63 | 5006.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5005.50 | 5015.21 | 5006.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 5019.80 | 5015.21 | 5006.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5007.40 | 5013.65 | 5006.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 5007.40 | 5013.65 | 5006.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 5010.00 | 5012.92 | 5006.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 5001.00 | 5012.92 | 5006.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 5006.90 | 5011.71 | 5006.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 5009.80 | 5011.71 | 5006.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 5016.00 | 5012.57 | 5007.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 4991.00 | 5012.57 | 5007.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 5008.70 | 5011.80 | 5007.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:15:00 | 4996.00 | 5011.80 | 5007.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 4978.30 | 5005.10 | 5004.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 4978.30 | 5005.10 | 5004.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 4980.10 | 5000.10 | 5002.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 4976.10 | 4992.26 | 4998.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 5018.00 | 4997.41 | 5000.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 5018.00 | 4997.41 | 5000.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 5018.00 | 4997.41 | 5000.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 5018.00 | 4997.41 | 5000.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 5034.00 | 5004.73 | 5003.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 15:15:00 | 5037.00 | 5022.52 | 5016.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 5022.00 | 5030.43 | 5023.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 5022.00 | 5030.43 | 5023.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 5022.00 | 5030.43 | 5023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 5021.30 | 5030.43 | 5023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 5005.10 | 5025.36 | 5021.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 5005.10 | 5025.36 | 5021.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5055.90 | 5031.47 | 5024.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:30:00 | 5090.00 | 5051.41 | 5036.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 5087.00 | 5089.77 | 5078.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 5083.90 | 5086.95 | 5078.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 13:15:00 | 5083.70 | 5082.01 | 5077.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 5129.90 | 5092.02 | 5082.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 5061.40 | 5092.02 | 5082.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 5053.90 | 5084.40 | 5080.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 5054.00 | 5084.40 | 5080.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 5029.50 | 5073.42 | 5075.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 5029.50 | 5073.42 | 5075.70 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 5116.00 | 5070.65 | 5070.39 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 5037.60 | 5070.24 | 5071.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 5030.80 | 5054.94 | 5063.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 5055.60 | 5051.21 | 5058.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 5055.60 | 5051.21 | 5058.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 5055.60 | 5051.21 | 5058.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:45:00 | 5069.80 | 5051.21 | 5058.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 5052.50 | 5051.47 | 5058.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 5064.80 | 5051.47 | 5058.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 5059.90 | 5053.15 | 5058.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 5028.40 | 5053.15 | 5058.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5017.50 | 5046.02 | 5054.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 5011.20 | 5036.70 | 5049.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 4999.40 | 5036.70 | 5049.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 4952.40 | 4902.28 | 4898.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 4952.40 | 4902.28 | 4898.98 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 4858.80 | 4907.40 | 4909.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 4857.90 | 4889.73 | 4900.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 11:15:00 | 4857.00 | 4849.45 | 4871.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:30:00 | 4854.80 | 4849.45 | 4871.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 4859.20 | 4852.15 | 4868.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:45:00 | 4860.50 | 4852.15 | 4868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 4856.50 | 4853.02 | 4867.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 4883.20 | 4853.02 | 4867.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 4868.90 | 4856.19 | 4867.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 4800.00 | 4856.19 | 4867.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 4840.00 | 4843.34 | 4843.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 4598.00 | 4659.85 | 4698.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 4776.60 | 4652.02 | 4669.49 | SL hit (close>ema200) qty=0.50 sl=4652.02 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 4727.40 | 4687.40 | 4683.12 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 4640.00 | 4684.62 | 4687.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 4626.70 | 4673.03 | 4681.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 4624.00 | 4615.69 | 4636.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 4651.10 | 4615.69 | 4636.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4667.20 | 4626.00 | 4639.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 4689.60 | 4626.00 | 4639.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 4660.70 | 4632.94 | 4641.06 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 4674.00 | 4650.28 | 4647.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 4745.00 | 4669.22 | 4656.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 4606.00 | 4656.58 | 4651.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 4606.00 | 4656.58 | 4651.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 4606.00 | 4656.58 | 4651.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 4606.00 | 4656.58 | 4651.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 4604.40 | 4646.14 | 4647.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 13:15:00 | 4590.80 | 4624.58 | 4636.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 4622.50 | 4588.29 | 4603.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 10:15:00 | 4622.50 | 4588.29 | 4603.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 4622.50 | 4588.29 | 4603.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 4622.50 | 4588.29 | 4603.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 4575.70 | 4585.77 | 4600.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 4552.30 | 4585.77 | 4600.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 4634.90 | 4574.83 | 4566.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 4634.90 | 4574.83 | 4566.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 4650.00 | 4589.87 | 4574.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 4700.00 | 4703.44 | 4671.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:30:00 | 4688.30 | 4703.44 | 4671.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4642.00 | 4688.48 | 4674.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 4640.80 | 4688.48 | 4674.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 4616.00 | 4673.99 | 4668.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 4616.00 | 4673.99 | 4668.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 4645.50 | 4661.70 | 4663.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 4594.50 | 4648.26 | 4657.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 4666.50 | 4636.32 | 4648.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 4666.50 | 4636.32 | 4648.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4666.50 | 4636.32 | 4648.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 4666.50 | 4636.32 | 4648.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 4683.80 | 4645.82 | 4651.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 4683.80 | 4645.82 | 4651.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 4693.20 | 4659.80 | 4657.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 4719.50 | 4671.74 | 4662.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 5120.80 | 5122.99 | 5026.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:15:00 | 5123.70 | 5122.99 | 5026.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 5034.50 | 5106.57 | 5055.69 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 5018.90 | 5041.43 | 5043.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 13:15:00 | 5001.20 | 5029.65 | 5037.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 5026.10 | 5020.54 | 5029.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 11:15:00 | 5026.10 | 5020.54 | 5029.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 5026.10 | 5020.54 | 5029.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 5022.00 | 5020.54 | 5029.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 5060.20 | 5028.47 | 5032.07 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 5070.00 | 5036.78 | 5035.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 5119.90 | 5054.07 | 5043.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 5095.00 | 5114.61 | 5086.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 5095.00 | 5114.61 | 5086.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 5095.00 | 5114.61 | 5086.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 5088.80 | 5114.61 | 5086.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 5077.00 | 5107.09 | 5085.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 5079.30 | 5107.09 | 5085.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 5062.10 | 5098.09 | 5083.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 5061.30 | 5098.09 | 5083.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 5020.50 | 5072.38 | 5073.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 4990.00 | 5055.91 | 5066.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 12:15:00 | 5024.20 | 4981.88 | 5002.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 12:15:00 | 5024.20 | 4981.88 | 5002.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 5024.20 | 4981.88 | 5002.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 5024.20 | 4981.88 | 5002.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 4985.00 | 4982.50 | 5001.31 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 5066.60 | 5014.01 | 5013.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 5105.50 | 5032.31 | 5021.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 5100.60 | 5125.42 | 5099.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 5100.60 | 5125.42 | 5099.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 5100.60 | 5125.42 | 5099.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 5100.60 | 5125.42 | 5099.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 5092.70 | 5118.87 | 5099.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 5086.20 | 5118.87 | 5099.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 5080.20 | 5111.14 | 5097.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 5073.70 | 5111.14 | 5097.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 5082.60 | 5101.33 | 5094.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:30:00 | 5086.00 | 5101.33 | 5094.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 5085.00 | 5098.06 | 5094.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 5096.90 | 5098.06 | 5094.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 5121.80 | 5102.81 | 5096.59 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 5068.00 | 5093.37 | 5094.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 5055.00 | 5085.69 | 5090.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 4877.50 | 4872.34 | 4928.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 4881.00 | 4872.34 | 4928.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 4879.00 | 4837.40 | 4878.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 4870.50 | 4837.40 | 4878.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 4853.50 | 4840.62 | 4875.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 4860.50 | 4840.62 | 4875.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 4868.50 | 4850.18 | 4874.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 4880.00 | 4850.18 | 4874.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 4863.00 | 4852.74 | 4873.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 4848.00 | 4851.19 | 4870.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 4605.60 | 4672.65 | 4712.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 4699.00 | 4655.01 | 4683.26 | SL hit (close>ema200) qty=0.50 sl=4655.01 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 4730.00 | 4691.59 | 4687.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 4785.00 | 4727.74 | 4709.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4724.50 | 4780.37 | 4753.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4724.50 | 4780.37 | 4753.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4724.50 | 4780.37 | 4753.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 4711.50 | 4780.37 | 4753.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 4764.00 | 4777.10 | 4754.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 4723.50 | 4777.10 | 4754.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 4733.50 | 4768.38 | 4752.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 4733.50 | 4768.38 | 4752.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 4722.00 | 4759.10 | 4749.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 4722.00 | 4759.10 | 4749.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 4720.00 | 4741.06 | 4742.91 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 4760.50 | 4744.95 | 4744.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 4796.00 | 4755.97 | 4749.64 | Break + close above crossover candle high |

### Cycle 58 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 4652.50 | 4743.56 | 4747.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 4625.50 | 4719.94 | 4736.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 4647.50 | 4625.58 | 4672.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 4647.50 | 4625.58 | 4672.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4647.50 | 4625.58 | 4672.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 4682.50 | 4625.58 | 4672.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 4672.50 | 4639.13 | 4667.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 4687.00 | 4639.13 | 4667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 4673.00 | 4645.91 | 4668.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 4676.50 | 4645.91 | 4668.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 4671.50 | 4651.02 | 4668.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 4671.50 | 4651.02 | 4668.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 4674.50 | 4655.72 | 4668.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 4740.50 | 4655.72 | 4668.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 4759.00 | 4676.38 | 4677.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 4746.50 | 4676.38 | 4677.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 4741.00 | 4689.30 | 4682.89 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4649.50 | 4699.19 | 4699.87 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 12:15:00 | 4724.50 | 4700.13 | 4699.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 13:15:00 | 4732.50 | 4706.60 | 4702.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 4764.50 | 4797.76 | 4765.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 4764.50 | 4797.76 | 4765.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 4764.50 | 4797.76 | 4765.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 4764.50 | 4797.76 | 4765.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 4782.40 | 4794.69 | 4767.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:00:00 | 4807.60 | 4792.68 | 4770.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 4804.90 | 4831.93 | 4800.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 4804.40 | 4824.38 | 4811.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 4773.10 | 4801.65 | 4803.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 4773.10 | 4801.65 | 4803.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 13:15:00 | 4770.10 | 4795.34 | 4800.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4841.80 | 4799.62 | 4800.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 4841.80 | 4799.62 | 4800.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 4841.80 | 4799.62 | 4800.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 4850.00 | 4799.62 | 4800.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 4851.60 | 4810.02 | 4805.06 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 13:15:00 | 4739.80 | 4798.90 | 4806.09 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 12:15:00 | 4825.00 | 4807.35 | 4805.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 13:15:00 | 4852.70 | 4816.42 | 4809.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 4820.30 | 4831.07 | 4819.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 4820.30 | 4831.07 | 4819.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4820.30 | 4831.07 | 4819.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4843.60 | 4834.08 | 4821.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 4886.30 | 4906.25 | 4908.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 4886.30 | 4906.25 | 4908.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 4855.70 | 4896.14 | 4904.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4822.60 | 4807.54 | 4838.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 10:00:00 | 4822.60 | 4807.54 | 4838.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 4708.50 | 4772.98 | 4804.66 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 4807.20 | 4776.56 | 4773.74 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 4771.10 | 4787.92 | 4788.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 4758.10 | 4778.70 | 4783.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 4758.00 | 4737.38 | 4753.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 4758.00 | 4737.38 | 4753.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4758.00 | 4737.38 | 4753.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 4774.00 | 4737.38 | 4753.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 4740.00 | 4737.90 | 4752.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 4740.60 | 4737.90 | 4752.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 4719.90 | 4699.13 | 4717.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 4719.90 | 4699.13 | 4717.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 4729.80 | 4705.26 | 4718.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 4729.80 | 4705.26 | 4718.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 4713.30 | 4706.87 | 4718.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 4695.00 | 4706.87 | 4718.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 4775.60 | 4718.72 | 4721.54 | SL hit (close>static) qty=1.00 sl=4733.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 4775.00 | 4729.97 | 4726.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 4840.00 | 4771.87 | 4758.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 4793.00 | 4803.31 | 4784.32 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-12 09:15:00 | 5870.00 | 2025-06-16 09:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-13 10:15:00 | 5900.50 | 2025-06-18 10:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-13 14:00:00 | 5887.50 | 2025-06-18 10:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-06-13 14:30:00 | 5889.00 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-13 15:15:00 | 5889.00 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-17 09:15:00 | 5953.50 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-06-18 09:15:00 | 5897.50 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-06-20 11:30:00 | 5694.50 | 2025-06-26 12:15:00 | 5650.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-07-03 13:45:00 | 5777.50 | 2025-07-09 09:15:00 | 5669.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-07-18 11:30:00 | 5318.50 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-21 11:15:00 | 5325.50 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-07-21 13:45:00 | 5331.00 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-21 15:15:00 | 5330.00 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-01 11:30:00 | 5096.50 | 2025-08-11 15:15:00 | 5098.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-08-22 09:30:00 | 5730.00 | 2025-08-22 15:15:00 | 5641.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-09-02 11:15:00 | 5132.50 | 2025-09-03 11:15:00 | 5187.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-03 14:45:00 | 5143.00 | 2025-09-04 09:15:00 | 5214.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-10-07 10:15:00 | 5240.00 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-07 14:00:00 | 5236.00 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-07 14:45:00 | 5237.50 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-08 09:30:00 | 5244.50 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-10-09 09:15:00 | 5231.50 | 2025-10-15 14:15:00 | 5329.50 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2025-10-20 09:15:00 | 5325.00 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-10-20 10:00:00 | 5328.00 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-20 14:45:00 | 5322.50 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-21 14:30:00 | 5310.00 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-29 11:00:00 | 5270.00 | 2025-10-30 14:15:00 | 5325.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-29 14:00:00 | 5269.50 | 2025-10-30 14:15:00 | 5325.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-29 15:00:00 | 5264.50 | 2025-10-30 14:15:00 | 5325.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-10-31 09:45:00 | 5263.50 | 2025-11-10 13:15:00 | 5118.50 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2025-11-04 09:45:00 | 5163.50 | 2025-11-10 13:15:00 | 5118.50 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-11-19 14:15:00 | 5020.50 | 2025-11-26 10:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-11-20 15:00:00 | 5025.00 | 2025-11-26 10:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-11-27 09:15:00 | 5048.00 | 2025-11-27 09:15:00 | 4982.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-02 13:00:00 | 4963.00 | 2025-12-04 10:15:00 | 4999.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-03 09:15:00 | 4958.00 | 2025-12-04 10:15:00 | 4999.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-03 15:15:00 | 4965.70 | 2025-12-04 10:15:00 | 4999.60 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-18 10:30:00 | 5090.00 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-22 09:30:00 | 5087.00 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-22 10:30:00 | 5083.90 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-22 13:15:00 | 5083.70 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-12-29 10:30:00 | 5011.20 | 2026-01-07 09:15:00 | 4952.40 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-12-29 11:00:00 | 4999.40 | 2026-01-07 09:15:00 | 4952.40 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2026-01-12 09:15:00 | 4800.00 | 2026-01-21 09:15:00 | 4598.00 | PARTIAL | 0.50 | 4.21% |
| SELL | retest2 | 2026-01-12 09:15:00 | 4800.00 | 2026-01-22 09:15:00 | 4776.60 | STOP_HIT | 0.50 | 0.49% |
| SELL | retest2 | 2026-01-14 09:15:00 | 4840.00 | 2026-01-22 12:15:00 | 4727.40 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2026-02-01 12:15:00 | 4552.30 | 2026-02-03 11:15:00 | 4634.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-06 14:45:00 | 4848.00 | 2026-03-13 10:15:00 | 4605.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 4848.00 | 2026-03-16 09:15:00 | 4699.00 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2026-04-02 13:00:00 | 4807.60 | 2026-04-07 12:15:00 | 4773.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-06 10:00:00 | 4804.90 | 2026-04-07 12:15:00 | 4773.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2026-04-07 10:15:00 | 4804.40 | 2026-04-07 12:15:00 | 4773.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4843.60 | 2026-04-21 10:15:00 | 4886.30 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2026-05-05 15:15:00 | 4695.00 | 2026-05-06 09:15:00 | 4775.60 | STOP_HIT | 1.00 | -1.72% |
