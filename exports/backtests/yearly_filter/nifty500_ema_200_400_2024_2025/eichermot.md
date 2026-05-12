# Eicher Motors Ltd. (EICHERMOT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 7309.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 8 |
| ALERT2_SKIP | 5 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 4 |
| TARGET_HIT | 9 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 20
- **Target hits / Stop hits / Partials:** 9 / 21 / 4
- **Avg / median % per leg:** 2.60% / -0.11%
- **Sum % (uncompounded):** 88.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 9 | 45.0% | 9 | 11 | 0 | 3.86% | 77.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 9 | 45.0% | 9 | 11 | 0 | 3.86% | 77.1% |
| SELL (all) | 14 | 5 | 35.7% | 0 | 10 | 4 | 0.80% | 11.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 5 | 35.7% | 0 | 10 | 4 | 0.80% | 11.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 14 | 41.2% | 9 | 21 | 4 | 2.60% | 88.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 4675.00 | 4805.59 | 4806.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 4567.35 | 4803.22 | 4805.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4843.25 | 4791.25 | 4798.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 4843.25 | 4791.25 | 4798.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 4843.25 | 4791.25 | 4798.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 4842.45 | 4791.25 | 4798.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 4795.70 | 4791.29 | 4798.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 4793.65 | 4791.45 | 4798.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 4770.70 | 4791.87 | 4798.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:30:00 | 4794.00 | 4791.99 | 4798.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 13:00:00 | 4790.80 | 4791.98 | 4798.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 4786.75 | 4791.93 | 4798.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:45:00 | 4797.60 | 4791.93 | 4798.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4553.97 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4532.16 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4554.30 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4551.26 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 4735.30 | 4755.89 | 4778.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 4823.00 | 4756.50 | 4778.37 | SL hit (close>ema200) qty=0.50 sl=4756.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 4939.50 | 4797.53 | 4796.98 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 4583.25 | 4796.25 | 4796.87 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 4897.40 | 4797.27 | 4797.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 4975.00 | 4802.32 | 4799.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4820.80 | 4851.28 | 4827.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 4820.80 | 4851.28 | 4827.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4820.80 | 4851.28 | 4827.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 4830.10 | 4851.28 | 4827.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 4835.85 | 4851.12 | 4827.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 4819.65 | 4851.12 | 4827.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 4817.15 | 4850.78 | 4827.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 4817.15 | 4850.78 | 4827.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 4835.50 | 4850.63 | 4827.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 4835.50 | 4850.63 | 4827.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 4796.00 | 4850.09 | 4827.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 4796.00 | 4850.09 | 4827.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 4849.00 | 4850.08 | 4827.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:00:00 | 4857.75 | 4846.06 | 4827.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 4852.15 | 4846.09 | 4827.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 4874.15 | 4845.67 | 4827.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 4787.00 | 4844.48 | 4827.10 | SL hit (close<static) qty=1.00 sl=4793.80 alert=retest2 |

### Cycle 5 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 4750.10 | 4818.14 | 4818.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 4724.15 | 4817.20 | 4817.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 4935.45 | 4810.36 | 4814.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 4935.45 | 4810.36 | 4814.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 4935.45 | 4810.36 | 4814.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 4935.45 | 4810.36 | 4814.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 4883.95 | 4811.10 | 4814.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 4879.20 | 4811.10 | 4814.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:00:00 | 4880.35 | 4812.81 | 4815.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 4876.50 | 4813.44 | 4815.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:30:00 | 4881.90 | 4815.46 | 4816.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 4881.75 | 4817.60 | 4817.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 4881.75 | 4817.60 | 4817.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 4897.40 | 4821.83 | 4819.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.93 | 4899.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 11:00:00 | 4943.40 | 4960.93 | 4899.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5184.26 | 5069.13 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 14:15:00 | 4905.00 | 4993.00 | 4993.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 09:15:00 | 4847.00 | 4990.64 | 4992.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4986.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4986.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4986.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 4986.40 | 4980.63 | 4986.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 4970.00 | 4980.52 | 4986.88 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 14:15:00 | 5099.00 | 4993.25 | 4992.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 15:15:00 | 5130.00 | 4994.61 | 4993.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 4996.50 | 4997.07 | 4994.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 14:15:00 | 4996.50 | 4997.07 | 4994.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 4996.50 | 4997.07 | 4994.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 4975.05 | 4997.07 | 4994.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 4972.50 | 4996.82 | 4994.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 4940.00 | 4996.82 | 4994.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 4975.50 | 4996.61 | 4994.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 4954.60 | 4996.61 | 4994.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 4968.55 | 4996.33 | 4994.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:45:00 | 4970.35 | 4996.33 | 4994.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 4987.15 | 4996.02 | 4994.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 4980.90 | 4996.02 | 4994.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 4983.40 | 4995.90 | 4994.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 4983.40 | 4995.90 | 4994.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 5005.00 | 4995.99 | 4994.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 4971.30 | 4995.99 | 4994.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 4969.00 | 4995.72 | 4994.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 4971.30 | 4995.72 | 4994.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 4999.75 | 4995.76 | 4994.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 5020.25 | 4995.85 | 4994.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 5013.00 | 4996.95 | 4994.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 5010.25 | 4997.08 | 4995.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 5023.25 | 4997.41 | 4995.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 5046.30 | 4998.28 | 4995.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 5079.40 | 5001.44 | 4997.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-01 09:15:00 | 5522.28 | 5153.58 | 5084.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 6854.00 | 7352.51 | 7353.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.07 | 7342.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.48 | 7177.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:00:00 | 7072.00 | 7049.48 | 7177.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 7060.00 | 7052.41 | 7176.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 7026.00 | 7123.00 | 7172.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:00:00 | 7009.50 | 7121.87 | 7171.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 7213.50 | 7120.51 | 7169.49 | SL hit (close>static) qty=1.00 sl=7180.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:45:00 | 4518.95 | 2024-06-18 09:15:00 | 4968.59 | TARGET_HIT | 1.00 | 9.95% |
| BUY | retest2 | 2024-06-04 15:15:00 | 4548.95 | 2024-06-18 10:15:00 | 4970.85 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2024-06-05 09:45:00 | 4516.90 | 2024-07-26 09:15:00 | 5003.85 | TARGET_HIT | 1.00 | 10.78% |
| SELL | retest2 | 2024-10-21 12:30:00 | 4793.65 | 2024-10-25 10:15:00 | 4553.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 4770.70 | 2024-10-25 10:15:00 | 4532.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 11:30:00 | 4794.00 | 2024-10-25 10:15:00 | 4554.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 13:00:00 | 4790.80 | 2024-10-25 10:15:00 | 4551.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 4793.65 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2024-10-22 10:00:00 | 4770.70 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -1.10% |
| SELL | retest2 | 2024-10-22 11:30:00 | 4794.00 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -0.60% |
| SELL | retest2 | 2024-10-22 13:00:00 | 4790.80 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -0.67% |
| BUY | retest2 | 2024-12-03 10:00:00 | 4857.75 | 2024-12-04 11:15:00 | 4787.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-12-03 10:45:00 | 4852.15 | 2024-12-04 11:15:00 | 4787.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-12-04 09:15:00 | 4874.15 | 2024-12-04 11:15:00 | 4787.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-05 14:15:00 | 4864.30 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-06 09:15:00 | 4925.70 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-12-09 10:45:00 | 4853.05 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-09 11:30:00 | 4855.95 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-12-09 15:15:00 | 4857.40 | 2024-12-12 09:15:00 | 4780.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-12-13 13:30:00 | 4833.60 | 2024-12-16 10:15:00 | 4802.95 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-12-13 14:45:00 | 4842.50 | 2024-12-16 10:15:00 | 4802.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-12-16 14:15:00 | 4837.65 | 2024-12-17 10:15:00 | 4808.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-12-27 11:15:00 | 4879.20 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-12-27 14:00:00 | 4880.35 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-12-27 14:45:00 | 4876.50 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-12-30 10:30:00 | 4881.90 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-03-12 13:15:00 | 5020.25 | 2025-04-01 09:15:00 | 5522.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 09:45:00 | 5013.00 | 2025-04-01 09:15:00 | 5514.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 11:00:00 | 5010.25 | 2025-04-01 09:15:00 | 5511.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 14:00:00 | 5023.25 | 2025-04-15 14:15:00 | 5525.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 09:15:00 | 5079.40 | 2025-04-16 09:15:00 | 5587.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 15:15:00 | 5078.40 | 2025-04-16 09:15:00 | 5586.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:30:00 | 7026.00 | 2026-05-04 09:15:00 | 7213.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-04-30 11:00:00 | 7009.50 | 2026-05-04 09:15:00 | 7213.50 | STOP_HIT | 1.00 | -2.91% |
