# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4522.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 18 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 15 / 12
- **Target hits / Stop hits / Partials:** 5 / 15 / 7
- **Avg / median % per leg:** 2.35% / 0.76%
- **Sum % (uncompounded):** 63.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.58% | 3.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.58% | 3.5% |
| SELL (all) | 21 | 14 | 66.7% | 4 | 10 | 7 | 2.86% | 60.1% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 0 | 10 | 3 | 0.01% | 0.1% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 19 | 7 | 36.8% | 1 | 15 | 3 | 0.19% | 3.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 14:15:00 | 4030.00 | 4580.07 | 4582.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 15:15:00 | 4017.00 | 4574.47 | 4579.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4230.15 | 4200.79 | 4340.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:00:00 | 4230.15 | 4200.79 | 4340.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 4334.15 | 4209.41 | 4330.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 4321.80 | 4209.41 | 4330.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4329.00 | 4210.60 | 4330.62 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 4613.70 | 4383.39 | 4383.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 11:15:00 | 4658.95 | 4386.13 | 4384.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 4453.00 | 4458.59 | 4425.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:30:00 | 4448.00 | 4458.59 | 4425.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 4410.10 | 4458.76 | 4426.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 4410.10 | 4458.76 | 4426.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 4293.55 | 4457.12 | 4425.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 4293.55 | 4457.12 | 4425.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 4050.95 | 4398.40 | 4399.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 4015.05 | 4394.58 | 4397.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 4262.95 | 4239.13 | 4301.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 4262.95 | 4239.13 | 4301.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 4244.00 | 4240.51 | 4301.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:45:00 | 4206.05 | 4244.83 | 4300.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 15:15:00 | 4228.00 | 4244.70 | 4299.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 12:15:00 | 4313.65 | 4246.13 | 4299.34 | SL hit (close>static) qty=1.00 sl=4308.95 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 4513.05 | 4327.51 | 4327.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 09:15:00 | 4559.00 | 4331.44 | 4329.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 09:15:00 | 4319.15 | 4364.87 | 4347.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 4319.15 | 4364.87 | 4347.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 4319.15 | 4364.87 | 4347.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:00:00 | 4319.15 | 4364.87 | 4347.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 4308.00 | 4364.30 | 4347.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 11:00:00 | 4308.00 | 4364.30 | 4347.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 4386.95 | 4364.52 | 4347.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 12:30:00 | 4397.40 | 4365.26 | 4347.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-19 09:15:00 | 4837.14 | 4539.37 | 4454.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 5597.00 | 5728.73 | 5729.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 15:15:00 | 5591.00 | 5726.06 | 5727.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 5714.00 | 5700.91 | 5713.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 5714.00 | 5700.91 | 5713.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 5714.00 | 5700.91 | 5713.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 5707.00 | 5700.91 | 5713.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 5724.00 | 5701.14 | 5713.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 5715.00 | 5701.14 | 5713.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 5729.00 | 5701.42 | 5713.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 5714.00 | 5702.45 | 5714.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 5748.00 | 5703.29 | 5714.34 | SL hit (close>static) qty=1.00 sl=5744.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 5891.00 | 5724.45 | 5724.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 5905.00 | 5726.24 | 5725.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 5740.50 | 5764.65 | 5745.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 10:15:00 | 5740.50 | 5764.65 | 5745.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 5740.50 | 5764.65 | 5745.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 5740.50 | 5764.65 | 5745.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 5714.00 | 5764.15 | 5745.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 5705.00 | 5764.15 | 5745.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 5774.50 | 5763.36 | 5745.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 5812.00 | 5763.38 | 5745.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 5816.50 | 5769.89 | 5750.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:00:00 | 5776.50 | 5773.17 | 5752.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 5785.00 | 5773.19 | 5752.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 5734.00 | 5772.69 | 5752.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 5734.00 | 5772.69 | 5752.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 5725.50 | 5772.22 | 5752.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 5725.50 | 5772.22 | 5752.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 5715.50 | 5771.19 | 5752.44 | SL hit (close<static) qty=1.00 sl=5720.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 5604.50 | 5736.66 | 5736.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 5564.50 | 5734.95 | 5735.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 5734.50 | 5725.87 | 5731.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 13:15:00 | 5734.50 | 5725.87 | 5731.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 5734.50 | 5725.87 | 5731.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 5731.50 | 5725.87 | 5731.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 5778.50 | 5726.39 | 5731.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 5776.00 | 5726.39 | 5731.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 5925.00 | 5736.40 | 5736.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 5932.00 | 5738.34 | 5737.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 12:15:00 | 5752.50 | 5761.58 | 5749.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:00:00 | 5752.50 | 5761.58 | 5749.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 5740.50 | 5761.53 | 5749.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 5740.50 | 5761.53 | 5749.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 5758.50 | 5761.50 | 5749.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 5711.50 | 5761.50 | 5749.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 5720.00 | 5761.09 | 5749.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 5734.00 | 5759.94 | 5749.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 5683.00 | 5795.20 | 5772.41 | SL hit (close<static) qty=1.00 sl=5695.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 10:15:00 | 5298.50 | 5752.16 | 5752.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 5158.00 | 5727.21 | 5739.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4914.80 | 4877.93 | 5088.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 14:30:00 | 4801.60 | 4918.93 | 5037.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 4830.00 | 4915.88 | 5034.29 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:45:00 | 4834.00 | 4915.08 | 5033.30 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:00:00 | 4830.00 | 4912.00 | 5028.82 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4561.52 | 4901.46 | 5004.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4588.50 | 4901.46 | 5004.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4592.30 | 4901.46 | 5004.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4588.50 | 4901.46 | 5004.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-04 09:15:00 | 4321.44 | 4875.98 | 4987.93 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-30 13:45:00 | 4206.05 | 2025-01-31 12:15:00 | 4313.65 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-30 15:15:00 | 4228.00 | 2025-01-31 12:15:00 | 4313.65 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-02-12 09:30:00 | 4216.45 | 2025-02-12 14:15:00 | 4328.90 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-02-14 11:15:00 | 4221.50 | 2025-02-17 15:15:00 | 4312.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-02-18 09:15:00 | 4276.00 | 2025-02-18 13:15:00 | 4344.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-03-03 12:30:00 | 4397.40 | 2025-03-19 09:15:00 | 4837.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-10 10:15:00 | 5714.00 | 2025-10-10 11:15:00 | 5748.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-27 09:15:00 | 5812.00 | 2025-10-30 15:15:00 | 5715.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-28 15:00:00 | 5816.50 | 2025-10-30 15:15:00 | 5715.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-10-30 10:00:00 | 5776.50 | 2025-10-30 15:15:00 | 5715.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-30 10:30:00 | 5785.00 | 2025-10-30 15:15:00 | 5715.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-19 12:30:00 | 5734.00 | 2025-12-02 12:15:00 | 5683.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest1 | 2026-02-19 14:30:00 | 4801.60 | 2026-03-02 09:15:00 | 4561.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-20 10:45:00 | 4830.00 | 2026-03-02 09:15:00 | 4588.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-20 11:45:00 | 4834.00 | 2026-03-02 09:15:00 | 4592.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-23 10:00:00 | 4830.00 | 2026-03-02 09:15:00 | 4588.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-19 14:30:00 | 4801.60 | 2026-03-04 09:15:00 | 4321.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-02-20 10:45:00 | 4830.00 | 2026-03-04 09:15:00 | 4347.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-02-20 11:45:00 | 4834.00 | 2026-03-04 09:15:00 | 4350.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-02-23 10:00:00 | 4830.00 | 2026-03-04 09:15:00 | 4347.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 4401.60 | 2026-04-15 09:15:00 | 4643.00 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2026-04-24 10:45:00 | 4543.50 | 2026-04-30 09:15:00 | 4316.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 10:00:00 | 4541.50 | 2026-04-30 09:15:00 | 4314.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 4534.10 | 2026-04-30 09:15:00 | 4307.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 10:45:00 | 4543.50 | 2026-05-06 14:15:00 | 4507.00 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2026-04-27 10:00:00 | 4541.50 | 2026-05-06 14:15:00 | 4507.00 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2026-04-27 11:15:00 | 4534.10 | 2026-05-06 14:15:00 | 4507.00 | STOP_HIT | 0.50 | 0.60% |
