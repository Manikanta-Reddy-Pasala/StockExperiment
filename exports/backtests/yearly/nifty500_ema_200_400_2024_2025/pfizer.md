# Pfizer Ltd. (PFIZER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4793.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 41 |
| PARTIAL | 11 |
| TARGET_HIT | 12 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 23 / 25
- **Target hits / Stop hits / Partials:** 12 / 25 / 11
- **Avg / median % per leg:** 2.67% / -0.93%
- **Sum % (uncompounded):** 127.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | -0.30% | -3.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | -0.30% | -3.0% |
| SELL (all) | 38 | 22 | 57.9% | 11 | 16 | 11 | 3.45% | 130.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 22 | 57.9% | 11 | 16 | 11 | 3.45% | 130.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 23 | 47.9% | 12 | 25 | 11 | 2.67% | 127.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 4521.65 | 4308.63 | 4308.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 10:15:00 | 4670.05 | 4312.22 | 4309.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 4604.00 | 4679.98 | 4549.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 4604.00 | 4679.98 | 4549.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 4533.70 | 4665.31 | 4558.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 4533.70 | 4665.31 | 4558.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 4531.00 | 4663.98 | 4558.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 4599.85 | 4663.98 | 4558.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 4510.00 | 4660.82 | 4558.66 | SL hit (close<static) qty=1.00 sl=4515.25 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 5152.60 | 5583.98 | 5585.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 11:15:00 | 5115.00 | 5579.31 | 5583.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 5344.95 | 5332.63 | 5418.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 5344.95 | 5332.63 | 5418.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 5344.95 | 5332.63 | 5418.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 5290.35 | 5330.30 | 5411.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 11:15:00 | 5284.05 | 5333.99 | 5405.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 5288.60 | 5333.65 | 5404.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 5242.00 | 5331.95 | 5402.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 5025.83 | 5267.07 | 5355.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 5024.17 | 5267.07 | 5355.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 5019.85 | 5264.60 | 5354.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 4979.90 | 5250.72 | 5344.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-20 14:15:00 | 4761.32 | 5177.60 | 5290.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 13:15:00 | 4955.00 | 4296.42 | 4296.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 5046.20 | 4364.29 | 4331.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 5591.00 | 5591.60 | 5289.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 5591.00 | 5591.60 | 5289.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 5320.00 | 5531.58 | 5316.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 5318.50 | 5531.58 | 5316.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 5276.50 | 5529.04 | 5315.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 5276.50 | 5529.04 | 5315.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 5290.00 | 5526.66 | 5315.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 5290.00 | 5526.66 | 5315.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 5321.00 | 5522.60 | 5315.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 5344.50 | 5522.60 | 5315.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 5332.00 | 5517.45 | 5316.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 5267.50 | 5508.14 | 5316.64 | SL hit (close<static) qty=1.00 sl=5304.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 5137.00 | 5275.30 | 5275.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 5126.00 | 5263.94 | 5269.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 5149.00 | 5141.89 | 5196.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 5149.00 | 5141.89 | 5196.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 5149.00 | 5141.89 | 5196.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 5109.00 | 5141.89 | 5196.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 5123.00 | 5141.66 | 5195.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 13:15:00 | 5495.00 | 5188.33 | 5208.80 | SL hit (close>static) qty=1.00 sl=5442.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 5351.00 | 5227.94 | 5227.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 13:15:00 | 5372.50 | 5233.59 | 5230.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 5252.00 | 5258.43 | 5244.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 5252.00 | 5258.43 | 5244.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5236.50 | 5258.21 | 5244.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 5233.50 | 5258.21 | 5244.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5246.50 | 5258.09 | 5244.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 5228.00 | 5258.09 | 5244.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 5279.50 | 5258.31 | 5244.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 5293.50 | 5258.45 | 5244.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 5290.50 | 5259.25 | 5245.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 5295.00 | 5259.36 | 5245.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 5295.00 | 5259.59 | 5245.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 5239.00 | 5259.63 | 5246.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 5229.50 | 5259.63 | 5246.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 5246.00 | 5259.49 | 5246.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 5233.50 | 5259.49 | 5246.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 5235.00 | 5259.25 | 5246.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 5235.00 | 5259.25 | 5246.13 | SL hit (close<static) qty=1.00 sl=5237.50 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 5070.50 | 5234.84 | 5235.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 5064.00 | 5233.14 | 5234.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 5037.00 | 5036.89 | 5092.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 09:15:00 | 5037.10 | 5036.89 | 5092.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 5084.00 | 5037.80 | 5090.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 5087.90 | 5037.80 | 5090.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 5103.10 | 5038.45 | 5090.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 5103.10 | 5038.45 | 5090.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 5100.00 | 5039.06 | 5090.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 5105.60 | 5039.06 | 5090.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 5091.60 | 5039.58 | 5090.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 5082.70 | 5042.95 | 5091.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 5064.60 | 5044.77 | 5091.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:45:00 | 5074.10 | 5045.10 | 5091.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:45:00 | 5080.00 | 5045.40 | 5091.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 5085.60 | 5046.69 | 5090.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 5085.60 | 5046.69 | 5090.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 5129.90 | 5047.51 | 5091.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 5129.90 | 5047.51 | 5091.11 | SL hit (close>static) qty=1.00 sl=5108.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 5147.60 | 4923.19 | 4923.05 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 4737.50 | 4924.10 | 4924.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 4711.50 | 4898.40 | 4911.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 4842.50 | 4841.09 | 4877.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:45:00 | 4847.00 | 4841.09 | 4877.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 4838.70 | 4787.28 | 4838.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 4838.70 | 4787.28 | 4838.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 4845.40 | 4787.86 | 4838.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:45:00 | 4841.80 | 4787.86 | 4838.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 4854.90 | 4788.53 | 4838.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 4854.90 | 4788.53 | 4838.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 4820.00 | 4788.84 | 4838.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 4796.00 | 4788.84 | 4838.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 4896.50 | 4789.58 | 4837.42 | SL hit (close>static) qty=1.00 sl=4870.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-28 09:15:00 | 4599.85 | 2024-06-28 11:15:00 | 4510.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-06-28 15:15:00 | 4560.00 | 2024-07-10 12:15:00 | 5016.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-02 09:15:00 | 5290.35 | 2024-12-13 09:15:00 | 5025.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 11:15:00 | 5284.05 | 2024-12-13 09:15:00 | 5024.17 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2024-12-05 12:15:00 | 5288.60 | 2024-12-13 10:15:00 | 5019.85 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-12-06 09:15:00 | 5242.00 | 2024-12-16 09:15:00 | 4979.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 09:15:00 | 5290.35 | 2024-12-20 14:15:00 | 4761.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-05 11:15:00 | 5284.05 | 2024-12-20 14:15:00 | 4755.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-05 12:15:00 | 5288.60 | 2024-12-20 14:15:00 | 4759.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 09:15:00 | 5242.00 | 2024-12-20 14:15:00 | 4717.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-31 09:15:00 | 5224.15 | 2025-01-13 14:15:00 | 5062.88 | PARTIAL | 0.50 | 3.09% |
| SELL | retest2 | 2025-01-02 15:00:00 | 5329.35 | 2025-01-14 09:15:00 | 5059.60 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-01-09 14:00:00 | 5325.90 | 2025-01-14 09:15:00 | 5058.75 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-01-09 15:15:00 | 5325.00 | 2025-01-15 12:15:00 | 4962.94 | PARTIAL | 0.50 | 6.80% |
| SELL | retest2 | 2025-01-13 11:00:00 | 5196.00 | 2025-01-16 09:15:00 | 4936.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-31 09:15:00 | 5224.15 | 2025-01-22 12:15:00 | 4796.42 | TARGET_HIT | 0.50 | 8.19% |
| SELL | retest2 | 2025-01-02 15:00:00 | 5329.35 | 2025-01-22 12:15:00 | 4793.31 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2025-01-09 14:00:00 | 5325.90 | 2025-01-22 12:15:00 | 4792.50 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2025-01-09 15:15:00 | 5325.00 | 2025-01-24 15:15:00 | 4701.73 | TARGET_HIT | 0.50 | 11.70% |
| SELL | retest2 | 2025-01-13 11:00:00 | 5196.00 | 2025-01-27 09:15:00 | 4676.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-21 09:15:00 | 5344.50 | 2025-07-22 09:15:00 | 5267.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-21 11:30:00 | 5332.00 | 2025-07-22 09:15:00 | 5267.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-25 09:15:00 | 5408.00 | 2025-07-28 15:15:00 | 5265.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-07-28 12:30:00 | 5334.50 | 2025-07-28 15:15:00 | 5265.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-01 10:15:00 | 5109.00 | 2025-10-13 13:15:00 | 5495.00 | STOP_HIT | 1.00 | -7.56% |
| SELL | retest2 | 2025-10-01 11:45:00 | 5123.00 | 2025-10-13 13:15:00 | 5495.00 | STOP_HIT | 1.00 | -7.26% |
| BUY | retest2 | 2025-10-28 14:15:00 | 5293.50 | 2025-10-30 11:15:00 | 5235.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-29 09:30:00 | 5290.50 | 2025-10-30 11:15:00 | 5235.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-29 10:45:00 | 5295.00 | 2025-10-30 11:15:00 | 5235.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-29 13:15:00 | 5295.00 | 2025-10-30 11:15:00 | 5235.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-12-19 12:00:00 | 5082.70 | 2025-12-22 15:15:00 | 5129.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-22 09:15:00 | 5064.60 | 2025-12-22 15:15:00 | 5129.90 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-12-22 09:45:00 | 5074.10 | 2025-12-22 15:15:00 | 5129.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-22 10:45:00 | 5080.00 | 2025-12-22 15:15:00 | 5129.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-12-23 09:15:00 | 5061.40 | 2026-01-08 14:15:00 | 4808.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 12:30:00 | 5065.10 | 2026-01-08 14:15:00 | 4811.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 09:15:00 | 5061.40 | 2026-01-27 09:15:00 | 4555.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 12:30:00 | 5065.10 | 2026-01-27 09:15:00 | 4558.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 5062.00 | 2026-02-11 12:15:00 | 5167.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-02-13 09:15:00 | 5031.30 | 2026-02-18 11:15:00 | 5136.80 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-02 09:15:00 | 4796.00 | 2026-04-02 13:15:00 | 4896.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-06 09:15:00 | 4804.10 | 2026-04-10 14:15:00 | 4869.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-06 12:30:00 | 4809.00 | 2026-04-10 14:15:00 | 4869.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-06 13:00:00 | 4818.10 | 2026-04-10 14:15:00 | 4869.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-04-07 09:15:00 | 4789.60 | 2026-04-10 14:15:00 | 4869.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-04-09 09:15:00 | 4820.00 | 2026-04-13 13:15:00 | 4871.60 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-10 10:15:00 | 4818.00 | 2026-04-13 13:15:00 | 4871.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-04-10 12:45:00 | 4817.20 | 2026-04-13 13:15:00 | 4871.60 | STOP_HIT | 1.00 | -1.13% |
