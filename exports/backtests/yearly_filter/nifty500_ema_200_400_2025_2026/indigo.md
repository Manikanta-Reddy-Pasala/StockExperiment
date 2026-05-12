# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4522.00
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
| ALERT2_SKIP | 3 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 12 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 14 / 7
- **Target hits / Stop hits / Partials:** 4 / 10 / 7
- **Avg / median % per leg:** 3.07% / 5.00%
- **Sum % (uncompounded):** 64.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.31% | -6.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.31% | -6.5% |
| SELL (all) | 16 | 14 | 87.5% | 4 | 5 | 7 | 4.44% | 71.1% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 0 | 5 | 3 | 1.39% | 11.1% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 13 | 6 | 46.2% | 0 | 10 | 3 | 0.35% | 4.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-30 13:15:00)

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

### Cycle 2 — BUY (started 2025-10-16 09:15:00)

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

### Cycle 3 — SELL (started 2025-11-07 15:15:00)

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

### Cycle 4 — BUY (started 2025-11-13 11:15:00)

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

### Cycle 5 — SELL (started 2025-12-05 10:15:00)

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
