# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 5555.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 25 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 32 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 20
- **Target hits / Stop hits / Partials:** 1 / 37 / 6
- **Avg / median % per leg:** 1.10% / 0.97%
- **Sum % (uncompounded):** 48.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 9 | 47.4% | 1 | 18 | 0 | 0.15% | 2.8% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.73% | -3.6% |
| BUY @ 3rd Alert (retest2) | 14 | 8 | 57.1% | 1 | 13 | 0 | 0.46% | 6.5% |
| SELL (all) | 25 | 15 | 60.0% | 0 | 19 | 6 | 1.83% | 45.7% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.14% | 0.1% |
| SELL @ 3rd Alert (retest2) | 24 | 14 | 58.3% | 0 | 18 | 6 | 1.90% | 45.5% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 6 | 0 | -0.59% | -3.5% |
| retest2 (combined) | 38 | 22 | 57.9% | 1 | 31 | 6 | 1.37% | 52.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 5209.00 | 5098.46 | 5083.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 5307.00 | 5217.78 | 5173.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 12:15:00 | 5291.50 | 5293.88 | 5249.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:45:00 | 5267.50 | 5293.88 | 5249.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 5234.00 | 5277.84 | 5249.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 5234.00 | 5277.84 | 5249.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 5259.00 | 5274.08 | 5250.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 5229.50 | 5257.56 | 5245.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 5204.50 | 5246.95 | 5241.49 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 5139.00 | 5225.36 | 5232.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 12:15:00 | 5135.50 | 5207.39 | 5223.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 5254.50 | 5213.07 | 5222.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 5254.50 | 5213.07 | 5222.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 5254.50 | 5213.07 | 5222.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 5254.50 | 5213.07 | 5222.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 5208.00 | 5212.05 | 5221.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 5223.00 | 5212.05 | 5221.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 5234.50 | 5216.54 | 5222.77 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 5268.50 | 5226.93 | 5226.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 5287.50 | 5260.61 | 5246.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 5232.50 | 5262.17 | 5249.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 5232.50 | 5262.17 | 5249.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 5232.50 | 5262.17 | 5249.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 5232.50 | 5262.17 | 5249.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 5198.50 | 5249.44 | 5245.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 5198.50 | 5249.44 | 5245.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5189.00 | 5237.35 | 5239.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 5137.00 | 5217.28 | 5230.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 5157.00 | 5115.92 | 5143.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 5157.00 | 5115.92 | 5143.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5157.00 | 5115.92 | 5143.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 5157.00 | 5115.92 | 5143.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 5159.50 | 5124.64 | 5144.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 5159.00 | 5124.64 | 5144.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 5224.00 | 5144.51 | 5151.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 5224.00 | 5144.51 | 5151.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 5164.00 | 5148.41 | 5152.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:45:00 | 5124.00 | 5139.23 | 5148.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 5448.00 | 5192.66 | 5169.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 5448.00 | 5192.66 | 5169.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 5674.50 | 5580.41 | 5531.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 5805.50 | 5821.71 | 5741.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 5805.50 | 5821.71 | 5741.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 5768.50 | 5786.17 | 5761.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 5768.50 | 5786.17 | 5761.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 5767.50 | 5782.44 | 5762.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 5826.00 | 5773.77 | 5763.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 5944.00 | 5983.00 | 5985.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 5944.00 | 5983.00 | 5985.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 5903.00 | 5967.00 | 5977.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 5940.00 | 5930.43 | 5952.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 5940.00 | 5930.43 | 5952.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 5940.00 | 5930.43 | 5952.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 5896.50 | 5929.44 | 5949.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 5889.00 | 5917.75 | 5942.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 5832.00 | 5763.32 | 5759.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 14:15:00 | 5832.00 | 5763.32 | 5759.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 5893.00 | 5806.94 | 5781.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 5908.00 | 6002.47 | 5947.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 5908.00 | 6002.47 | 5947.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 5908.00 | 6002.47 | 5947.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 5890.50 | 6002.47 | 5947.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 5939.50 | 5989.88 | 5946.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 5954.00 | 5989.80 | 5950.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-11 09:15:00 | 6549.40 | 6404.14 | 6362.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 6455.00 | 6467.80 | 6467.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 6425.00 | 6459.24 | 6463.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 6499.00 | 6457.71 | 6461.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 6499.00 | 6457.71 | 6461.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 6499.00 | 6457.71 | 6461.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 6560.50 | 6457.71 | 6461.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 10:15:00 | 6578.50 | 6481.87 | 6472.47 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 6428.00 | 6481.31 | 6484.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 6404.00 | 6454.22 | 6470.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 6486.50 | 6460.68 | 6471.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 6486.50 | 6460.68 | 6471.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 6486.50 | 6460.68 | 6471.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 6473.00 | 6460.68 | 6471.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 6509.00 | 6470.34 | 6474.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 6459.00 | 6477.54 | 6477.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 6492.50 | 6480.53 | 6479.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 6492.50 | 6480.53 | 6479.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 6533.00 | 6491.77 | 6484.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 6585.50 | 6593.09 | 6555.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 12:00:00 | 6585.50 | 6593.09 | 6555.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6623.00 | 6612.67 | 6580.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 6620.00 | 6612.67 | 6580.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 6594.50 | 6609.03 | 6581.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 6583.50 | 6609.03 | 6581.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 6567.50 | 6600.73 | 6580.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 6567.50 | 6600.73 | 6580.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 6535.50 | 6587.68 | 6576.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 6535.50 | 6587.68 | 6576.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 6532.00 | 6576.55 | 6572.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 6532.00 | 6576.55 | 6572.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 6477.00 | 6556.64 | 6563.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 6458.00 | 6520.38 | 6541.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 6479.00 | 6477.47 | 6501.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 6479.00 | 6477.47 | 6501.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 6486.00 | 6479.50 | 6498.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 6624.50 | 6506.60 | 6508.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 6614.00 | 6528.08 | 6518.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 6642.00 | 6585.99 | 6554.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 6639.50 | 6673.52 | 6639.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 15:15:00 | 6639.50 | 6673.52 | 6639.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 6639.50 | 6673.52 | 6639.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 6741.50 | 6692.82 | 6651.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 6978.50 | 6995.24 | 6995.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 6978.50 | 6995.24 | 6995.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 13:15:00 | 6946.50 | 6985.49 | 6991.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 7000.50 | 6974.62 | 6983.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 7000.50 | 6974.62 | 6983.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 7000.50 | 6974.62 | 6983.72 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 7000.00 | 6988.92 | 6988.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 7250.50 | 7043.01 | 7013.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 10:15:00 | 7188.50 | 7237.02 | 7160.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:00:00 | 7188.50 | 7237.02 | 7160.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 7349.00 | 7415.25 | 7359.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 7346.50 | 7415.25 | 7359.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 7294.50 | 7391.10 | 7353.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 7274.00 | 7391.10 | 7353.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 7107.50 | 7295.64 | 7314.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 6960.00 | 7228.51 | 7282.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 6932.00 | 6925.82 | 7046.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 6932.00 | 6925.82 | 7046.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 6849.00 | 6916.41 | 6965.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 6720.00 | 6916.41 | 6965.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 7000.00 | 6888.92 | 6923.74 | SL hit (close>static) qty=1.00 sl=6972.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 6997.50 | 6942.20 | 6938.85 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 6902.00 | 6935.49 | 6936.47 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 7085.50 | 6965.49 | 6950.01 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 6936.00 | 6998.05 | 6998.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 11:15:00 | 6815.50 | 6951.45 | 6976.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 6706.00 | 6668.49 | 6720.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 10:15:00 | 6706.00 | 6668.49 | 6720.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 6706.00 | 6668.49 | 6720.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 6706.00 | 6668.49 | 6720.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 6710.00 | 6676.80 | 6719.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:30:00 | 6734.00 | 6676.80 | 6719.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 6748.50 | 6691.14 | 6722.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 6753.00 | 6691.14 | 6722.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6764.50 | 6705.81 | 6726.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6770.00 | 6705.81 | 6726.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 6820.00 | 6743.40 | 6740.80 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 6707.50 | 6736.22 | 6737.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 6688.00 | 6717.43 | 6727.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 6760.00 | 6722.27 | 6727.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 6760.00 | 6722.27 | 6727.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 6760.00 | 6722.27 | 6727.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 6681.00 | 6722.27 | 6727.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 6693.50 | 6567.62 | 6564.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 6693.50 | 6567.62 | 6564.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 6773.50 | 6660.12 | 6615.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 15:15:00 | 6771.00 | 6788.94 | 6742.54 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:15:00 | 6859.00 | 6788.94 | 6742.54 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:15:00 | 6815.00 | 6789.95 | 6747.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 6808.00 | 6792.26 | 6752.15 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 12:00:00 | 6803.00 | 6794.41 | 6756.78 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 6848.50 | 6844.61 | 6812.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 6832.00 | 6844.61 | 6812.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 6851.50 | 6861.84 | 6841.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 6851.50 | 6861.84 | 6841.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 6803.50 | 6850.17 | 6838.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 6803.50 | 6850.17 | 6838.40 | SL hit (close<ema400) qty=1.00 sl=6838.40 alert=retest1 |

### Cycle 24 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 6761.00 | 6830.31 | 6831.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 6745.00 | 6813.25 | 6823.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 6314.50 | 6289.36 | 6341.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 14:00:00 | 6314.50 | 6289.36 | 6341.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 6305.50 | 6297.32 | 6332.83 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 6352.00 | 6330.47 | 6328.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 6495.00 | 6365.22 | 6344.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 6435.50 | 6445.05 | 6402.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 6436.50 | 6445.05 | 6402.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 6446.00 | 6445.24 | 6406.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 6467.50 | 6445.24 | 6406.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 6462.50 | 6448.69 | 6411.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 6466.50 | 6441.64 | 6414.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 6465.00 | 6438.55 | 6417.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 6438.50 | 6438.54 | 6419.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 6493.00 | 6438.54 | 6419.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 6560.00 | 6618.11 | 6620.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 6560.00 | 6618.11 | 6620.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 6496.00 | 6593.69 | 6608.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 10:15:00 | 6598.00 | 6577.39 | 6595.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 6598.00 | 6577.39 | 6595.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 6598.00 | 6577.39 | 6595.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 6598.00 | 6577.39 | 6595.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 6600.00 | 6581.91 | 6596.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 6600.00 | 6581.91 | 6596.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 6605.00 | 6586.53 | 6597.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 6575.50 | 6584.33 | 6595.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 14:15:00 | 6246.72 | 6313.19 | 6374.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 6301.00 | 6293.47 | 6344.20 | SL hit (close>ema200) qty=0.50 sl=6293.47 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 6399.00 | 6365.57 | 6364.71 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 6353.50 | 6379.56 | 6379.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 13:15:00 | 6347.00 | 6373.05 | 6376.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 6238.00 | 6235.36 | 6276.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:45:00 | 6249.50 | 6235.36 | 6276.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 5984.50 | 6181.05 | 6241.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 5910.50 | 6125.74 | 6210.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 5614.97 | 5729.25 | 5830.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 5648.50 | 5647.67 | 5724.76 | SL hit (close>ema200) qty=0.50 sl=5647.67 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 13:15:00 | 5619.50 | 5578.56 | 5577.55 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 5512.00 | 5569.02 | 5575.59 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 5618.50 | 5572.99 | 5571.95 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 5486.00 | 5557.87 | 5567.06 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 5631.50 | 5565.68 | 5556.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 5665.50 | 5585.65 | 5566.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 5609.50 | 5619.59 | 5589.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 5609.50 | 5619.59 | 5589.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 5586.00 | 5612.87 | 5589.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 5597.00 | 5612.87 | 5589.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 5637.00 | 5617.70 | 5593.82 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 5567.50 | 5602.31 | 5604.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 5514.00 | 5584.65 | 5596.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 5574.00 | 5570.48 | 5584.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:15:00 | 5606.50 | 5570.48 | 5584.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 5607.00 | 5577.78 | 5586.73 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 5657.00 | 5594.02 | 5589.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 5760.00 | 5627.22 | 5604.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 5806.00 | 5808.40 | 5729.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:45:00 | 5807.00 | 5808.40 | 5729.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 5777.00 | 5804.56 | 5771.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 5771.00 | 5804.56 | 5771.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 5770.00 | 5797.65 | 5771.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 5769.50 | 5797.65 | 5771.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5778.00 | 5793.72 | 5772.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 5778.00 | 5793.72 | 5772.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5760.00 | 5786.97 | 5771.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 5760.00 | 5786.97 | 5771.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 5741.00 | 5777.78 | 5768.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 5702.50 | 5777.78 | 5768.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 5698.00 | 5761.82 | 5762.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 5674.50 | 5744.36 | 5754.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 5780.00 | 5733.45 | 5745.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 13:15:00 | 5780.00 | 5733.45 | 5745.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 5780.00 | 5733.45 | 5745.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 5782.50 | 5733.45 | 5745.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 5786.50 | 5744.06 | 5748.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 5786.50 | 5744.06 | 5748.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 5784.00 | 5752.05 | 5752.00 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 5694.00 | 5740.44 | 5746.73 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 5770.00 | 5752.56 | 5751.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 5797.50 | 5764.26 | 5756.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 5705.50 | 5757.75 | 5755.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 5705.50 | 5757.75 | 5755.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 5705.50 | 5757.75 | 5755.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 5703.00 | 5757.75 | 5755.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 5684.00 | 5743.00 | 5749.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 5669.50 | 5697.93 | 5718.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 5700.00 | 5671.55 | 5694.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 5700.00 | 5671.55 | 5694.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 5700.00 | 5671.55 | 5694.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 5704.50 | 5671.55 | 5694.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 5700.00 | 5677.24 | 5694.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 5700.00 | 5677.24 | 5694.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 5700.00 | 5681.79 | 5695.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 5705.50 | 5681.79 | 5695.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 5629.50 | 5664.81 | 5684.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 5660.50 | 5664.81 | 5684.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5570.00 | 5505.30 | 5559.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 5570.00 | 5505.30 | 5559.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5534.00 | 5511.04 | 5557.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 5501.00 | 5522.29 | 5551.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 5519.50 | 5512.32 | 5539.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 5613.00 | 5541.58 | 5537.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 5613.00 | 5541.58 | 5537.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 5675.50 | 5578.51 | 5555.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 5650.50 | 5663.74 | 5622.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:00:00 | 5650.50 | 5663.74 | 5622.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 5609.50 | 5648.81 | 5632.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 5609.50 | 5648.81 | 5632.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 5596.00 | 5638.24 | 5629.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 5590.00 | 5638.24 | 5629.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 5620.50 | 5630.61 | 5627.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:45:00 | 5616.50 | 5630.61 | 5627.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 5621.00 | 5628.69 | 5627.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 5634.00 | 5628.69 | 5627.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 5619.50 | 5626.85 | 5626.41 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 5606.00 | 5622.68 | 5624.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 5578.00 | 5613.74 | 5620.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 13:15:00 | 5607.50 | 5605.18 | 5614.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 14:00:00 | 5607.50 | 5605.18 | 5614.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5596.00 | 5603.34 | 5613.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 5596.00 | 5603.34 | 5613.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 5470.50 | 5422.46 | 5447.68 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 5556.50 | 5462.16 | 5462.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 13:15:00 | 5590.00 | 5505.86 | 5483.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 5663.50 | 5664.36 | 5605.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 5663.50 | 5664.36 | 5605.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 5615.00 | 5689.03 | 5666.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 5615.00 | 5689.03 | 5666.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 5577.50 | 5666.72 | 5658.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 5577.50 | 5666.72 | 5658.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 5532.00 | 5639.78 | 5647.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 5498.50 | 5611.52 | 5633.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 5571.50 | 5554.92 | 5591.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:45:00 | 5572.00 | 5554.92 | 5591.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 5543.50 | 5556.62 | 5581.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 5576.00 | 5556.62 | 5581.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 5508.50 | 5500.93 | 5530.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 5508.50 | 5500.93 | 5530.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 5543.50 | 5509.45 | 5531.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 5558.50 | 5509.45 | 5531.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 5555.50 | 5518.66 | 5533.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 5555.50 | 5518.66 | 5533.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 5550.00 | 5524.93 | 5534.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:15:00 | 5569.50 | 5524.93 | 5534.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 5567.00 | 5542.95 | 5542.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 5688.00 | 5577.89 | 5558.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 5717.00 | 5717.25 | 5656.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 5678.50 | 5704.51 | 5665.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 5678.50 | 5704.51 | 5665.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 5676.50 | 5704.51 | 5665.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 5675.50 | 5698.71 | 5666.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 5675.50 | 5698.71 | 5666.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 5748.00 | 5708.57 | 5673.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 5783.50 | 5715.86 | 5679.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 5718.50 | 5834.02 | 5842.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 5718.50 | 5834.02 | 5842.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 15:15:00 | 5700.00 | 5807.22 | 5829.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 5694.50 | 5645.99 | 5709.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 5694.50 | 5645.99 | 5709.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5694.50 | 5645.99 | 5709.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 5694.50 | 5645.99 | 5709.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 5718.50 | 5660.50 | 5710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 5700.50 | 5660.50 | 5710.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 5762.50 | 5680.90 | 5714.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:45:00 | 5769.50 | 5680.90 | 5714.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 5782.00 | 5737.22 | 5734.14 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 5699.50 | 5729.68 | 5730.99 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 5749.50 | 5733.64 | 5732.67 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 5678.00 | 5723.46 | 5728.72 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 5875.00 | 5758.26 | 5743.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 5907.50 | 5788.10 | 5758.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 5786.00 | 5836.46 | 5800.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 5786.00 | 5836.46 | 5800.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 5786.00 | 5836.46 | 5800.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 6078.50 | 5801.09 | 5795.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 5889.50 | 5864.34 | 5833.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 5722.50 | 5844.66 | 5836.57 | SL hit (close<static) qty=1.00 sl=5754.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 5620.00 | 5799.73 | 5816.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 09:15:00 | 5525.50 | 5667.09 | 5736.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 15:15:00 | 5551.00 | 5536.13 | 5594.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 5551.00 | 5536.13 | 5594.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 5551.00 | 5536.13 | 5594.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 5431.50 | 5536.13 | 5594.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 5474.50 | 5491.80 | 5545.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 5662.50 | 5523.21 | 5545.79 | SL hit (close>static) qty=1.00 sl=5649.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 5659.00 | 5574.42 | 5566.64 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 5535.00 | 5570.66 | 5575.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 5515.50 | 5559.62 | 5569.96 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 5667.50 | 5579.66 | 5577.18 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 5530.50 | 5568.58 | 5572.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 5525.00 | 5553.69 | 5565.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 5493.00 | 5488.65 | 5516.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 5493.00 | 5488.65 | 5516.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 5493.00 | 5488.65 | 5516.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 5470.50 | 5488.65 | 5516.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 5465.00 | 5472.96 | 5493.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 5483.50 | 5472.96 | 5493.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 5555.00 | 5489.37 | 5499.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:45:00 | 5581.00 | 5489.37 | 5499.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 5538.50 | 5499.19 | 5503.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 5559.50 | 5499.19 | 5503.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 5624.00 | 5524.15 | 5514.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 5657.50 | 5603.78 | 5563.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 5657.00 | 5669.42 | 5625.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 5657.00 | 5669.42 | 5625.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 5724.00 | 5705.66 | 5678.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:30:00 | 5704.50 | 5705.66 | 5678.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 5832.50 | 5875.75 | 5834.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 5832.50 | 5875.75 | 5834.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 5810.00 | 5862.60 | 5831.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 5869.00 | 5862.60 | 5831.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 5865.50 | 5863.18 | 5834.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 5889.50 | 5865.12 | 5840.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 5907.00 | 5873.09 | 5846.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:00:00 | 5889.50 | 5876.37 | 5850.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 5774.50 | 5851.36 | 5845.16 | SL hit (close<static) qty=1.00 sl=5809.50 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 5754.00 | 5831.89 | 5836.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 5682.50 | 5789.16 | 5812.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 5633.00 | 5614.60 | 5677.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 5633.00 | 5614.60 | 5677.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 5680.00 | 5627.68 | 5677.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 5680.00 | 5627.68 | 5677.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 5675.00 | 5637.15 | 5677.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 5686.00 | 5637.15 | 5677.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 5741.00 | 5657.92 | 5683.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 5741.00 | 5657.92 | 5683.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 5787.00 | 5683.73 | 5692.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 5799.50 | 5683.73 | 5692.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 5785.50 | 5704.09 | 5701.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 5834.00 | 5730.07 | 5713.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 15:15:00 | 5872.50 | 5873.63 | 5820.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:15:00 | 5952.00 | 5873.63 | 5820.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 5867.00 | 5870.64 | 5836.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 5848.50 | 5870.64 | 5836.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 5796.50 | 5853.07 | 5834.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 5796.50 | 5853.07 | 5834.07 | SL hit (close<ema400) qty=1.00 sl=5834.07 alert=retest1 |

### Cycle 60 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 5775.00 | 5820.64 | 5821.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 5720.00 | 5779.73 | 5799.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 5771.00 | 5769.23 | 5785.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 5771.00 | 5769.23 | 5785.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 5771.00 | 5769.23 | 5785.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 5789.00 | 5769.23 | 5785.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 5768.00 | 5768.98 | 5784.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 5768.00 | 5768.98 | 5784.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 5807.50 | 5776.69 | 5786.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 5729.00 | 5776.69 | 5786.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 5735.50 | 5701.09 | 5697.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 5735.50 | 5701.09 | 5697.74 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 5650.00 | 5687.87 | 5692.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 5636.00 | 5677.50 | 5687.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 5550.50 | 5543.13 | 5578.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 5550.50 | 5543.13 | 5578.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5550.50 | 5543.13 | 5578.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:15:00 | 5507.00 | 5540.07 | 5568.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 5507.00 | 5533.65 | 5563.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:00:00 | 5516.00 | 5530.12 | 5558.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:15:00 | 5514.00 | 5524.86 | 5551.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5231.65 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5231.65 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5240.20 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5238.30 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 5105.00 | 5095.23 | 5187.49 | SL hit (close>ema200) qty=0.50 sl=5095.23 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 5201.50 | 5188.38 | 5187.10 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 5031.50 | 5158.86 | 5174.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 4966.50 | 5120.39 | 5155.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 5011.00 | 4996.85 | 5049.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:45:00 | 5008.50 | 4996.85 | 5049.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 5050.50 | 5007.58 | 5049.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 5081.00 | 5007.58 | 5049.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 5050.00 | 5016.06 | 5049.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 5093.50 | 5016.06 | 5049.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 5085.00 | 5043.12 | 5057.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 5085.50 | 5043.12 | 5057.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 5146.00 | 5076.48 | 5070.54 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 5035.00 | 5080.38 | 5084.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 5007.50 | 5058.14 | 5073.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 5008.50 | 5002.43 | 5027.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 5008.50 | 5002.43 | 5027.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 4854.00 | 4945.30 | 4990.21 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 5113.50 | 4989.26 | 4977.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 15:15:00 | 5120.00 | 5036.09 | 5001.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 5104.50 | 5214.50 | 5144.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 5104.50 | 5214.50 | 5144.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 5104.50 | 5214.50 | 5144.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 5104.50 | 5214.50 | 5144.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 5101.00 | 5191.80 | 5140.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 5101.00 | 5191.80 | 5140.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 5112.00 | 5175.84 | 5138.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 5113.00 | 5175.84 | 5138.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 5089.00 | 5143.43 | 5131.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 5089.00 | 5143.43 | 5131.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 5080.00 | 5130.74 | 5126.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 5059.50 | 5130.74 | 5126.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 5032.50 | 5111.10 | 5118.05 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 5139.00 | 5109.78 | 5109.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 5160.50 | 5124.68 | 5116.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 5060.50 | 5117.68 | 5116.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 5060.50 | 5117.68 | 5116.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 5060.50 | 5117.68 | 5116.59 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 5099.00 | 5113.94 | 5114.99 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 5162.50 | 5123.11 | 5118.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 5182.00 | 5134.89 | 5124.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 5216.00 | 5239.69 | 5198.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 5216.00 | 5239.69 | 5198.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5575.00 | 5636.62 | 5566.57 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 5441.00 | 5526.23 | 5532.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 5434.00 | 5507.79 | 5523.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 10:15:00 | 5506.00 | 5505.14 | 5519.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 10:15:00 | 5506.00 | 5505.14 | 5519.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 5506.00 | 5505.14 | 5519.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 5514.50 | 5505.14 | 5519.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 5493.50 | 5473.15 | 5494.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 5494.00 | 5473.15 | 5494.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 5507.00 | 5479.92 | 5495.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 5507.00 | 5479.92 | 5495.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 5444.50 | 5472.84 | 5490.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:45:00 | 5425.00 | 5470.87 | 5488.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 5526.50 | 5490.50 | 5492.91 | SL hit (close>static) qty=1.00 sl=5514.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 5510.00 | 5497.60 | 5495.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 5586.00 | 5534.26 | 5515.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 15:15:00 | 5877.50 | 5934.49 | 5847.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:15:00 | 5830.00 | 5934.49 | 5847.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 5846.00 | 5916.79 | 5847.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 5808.50 | 5916.79 | 5847.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 5818.00 | 5897.03 | 5844.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 5818.00 | 5897.03 | 5844.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 5797.00 | 5877.03 | 5840.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 5781.50 | 5877.03 | 5840.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 5640.50 | 5788.32 | 5807.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 5598.00 | 5750.26 | 5788.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 5517.00 | 5492.08 | 5553.29 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 13:45:00 | 5475.50 | 5494.98 | 5536.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 5468.00 | 5360.08 | 5407.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5468.00 | 5360.08 | 5407.62 | SL hit (close>ema400) qty=1.00 sl=5407.62 alert=retest1 |

### Cycle 75 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 5446.00 | 5373.29 | 5369.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 5525.50 | 5419.53 | 5392.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 5577.00 | 5581.47 | 5514.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:00:00 | 5577.00 | 5581.47 | 5514.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 13:45:00 | 5124.00 | 2025-05-26 09:15:00 | 5448.00 | STOP_HIT | 1.00 | -6.32% |
| BUY | retest2 | 2025-06-11 09:15:00 | 5826.00 | 2025-06-17 10:15:00 | 5944.00 | STOP_HIT | 1.00 | 2.03% |
| SELL | retest2 | 2025-06-18 10:30:00 | 5896.50 | 2025-06-24 14:15:00 | 5832.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-06-18 11:30:00 | 5889.00 | 2025-06-24 14:15:00 | 5832.00 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-06-27 11:30:00 | 5954.00 | 2025-07-11 09:15:00 | 6549.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 6459.00 | 2025-07-22 13:15:00 | 6492.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-04 09:45:00 | 6741.50 | 2025-08-13 12:15:00 | 6978.50 | STOP_HIT | 1.00 | 3.52% |
| SELL | retest2 | 2025-08-28 09:15:00 | 6720.00 | 2025-08-28 13:15:00 | 7000.00 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-09-10 10:15:00 | 6681.00 | 2025-09-16 11:15:00 | 6693.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-19 09:15:00 | 6859.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2025-09-19 10:15:00 | 6815.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-19 11:15:00 | 6808.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2025-09-19 12:00:00 | 6803.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-10-08 10:15:00 | 6467.50 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-10-08 11:00:00 | 6462.50 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2025-10-08 13:15:00 | 6466.50 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-10-08 14:30:00 | 6465.00 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-10-09 09:15:00 | 6493.00 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-10-16 14:00:00 | 6575.50 | 2025-10-24 14:15:00 | 6246.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 14:00:00 | 6575.50 | 2025-10-27 11:15:00 | 6301.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-11-03 10:45:00 | 5910.50 | 2025-11-07 09:15:00 | 5614.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:45:00 | 5910.50 | 2025-11-10 10:15:00 | 5648.50 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-12-10 10:45:00 | 5501.00 | 2025-12-11 14:15:00 | 5613.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-10 13:30:00 | 5519.50 | 2025-12-11 14:15:00 | 5613.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-01-07 09:15:00 | 5783.50 | 2026-01-09 14:15:00 | 5718.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-01-20 09:15:00 | 6078.50 | 2026-01-21 09:15:00 | 5722.50 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest2 | 2026-01-20 13:00:00 | 5889.50 | 2026-01-21 09:15:00 | 5722.50 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-01-27 09:15:00 | 5431.50 | 2026-01-28 09:15:00 | 5662.50 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2026-01-27 14:15:00 | 5474.50 | 2026-01-28 09:15:00 | 5662.50 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2026-02-11 11:30:00 | 5889.50 | 2026-02-12 09:15:00 | 5774.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-11 12:45:00 | 5907.00 | 2026-02-12 09:15:00 | 5774.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-02-11 14:00:00 | 5889.50 | 2026-02-12 09:15:00 | 5774.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2026-02-19 09:15:00 | 5952.00 | 2026-02-19 14:15:00 | 5796.50 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 5729.00 | 2026-02-26 14:15:00 | 5735.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-03-05 13:15:00 | 5507.00 | 2026-03-09 10:15:00 | 5231.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 5507.00 | 2026-03-09 10:15:00 | 5231.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:00:00 | 5516.00 | 2026-03-09 10:15:00 | 5240.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:15:00 | 5514.00 | 2026-03-09 10:15:00 | 5238.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:15:00 | 5507.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.30% |
| SELL | retest2 | 2026-03-05 13:45:00 | 5507.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.30% |
| SELL | retest2 | 2026-03-05 15:00:00 | 5516.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.45% |
| SELL | retest2 | 2026-03-06 10:15:00 | 5514.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.42% |
| SELL | retest2 | 2026-04-16 12:45:00 | 5425.00 | 2026-04-17 09:15:00 | 5526.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest1 | 2026-04-29 13:45:00 | 5475.50 | 2026-05-04 10:15:00 | 5468.00 | STOP_HIT | 1.00 | 0.14% |
