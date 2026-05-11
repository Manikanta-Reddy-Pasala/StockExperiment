# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4360.00
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
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 3 |
| TARGET_HIT | 12 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 12
- **Target hits / Stop hits / Partials:** 12 / 15 / 3
- **Avg / median % per leg:** 3.96% / 5.00%
- **Sum % (uncompounded):** 118.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 12 | 60.0% | 12 | 8 | 0 | 5.42% | 108.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 12 | 60.0% | 12 | 8 | 0 | 5.42% | 108.4% |
| SELL (all) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.04% | 10.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.04% | 10.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 18 | 60.0% | 12 | 15 | 3 | 3.96% | 118.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 5450.00 | 5686.91 | 5687.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 5408.00 | 5677.13 | 5682.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 4783.95 | 4763.94 | 4942.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 4783.95 | 4763.94 | 4942.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 4878.65 | 4774.68 | 4914.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 4905.00 | 4774.68 | 4914.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 4877.75 | 4780.40 | 4913.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:45:00 | 4867.40 | 4783.36 | 4913.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 4858.15 | 4786.29 | 4912.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 4832.15 | 4792.71 | 4911.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4624.03 | 4777.70 | 4891.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4615.24 | 4777.70 | 4891.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4590.54 | 4777.70 | 4891.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-06 12:15:00 | 4780.25 | 4763.85 | 4875.07 | SL hit (close>ema200) qty=0.50 sl=4763.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 5154.55 | 4936.40 | 4936.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 5181.20 | 4938.83 | 4937.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 15:15:00 | 5485.05 | 5500.86 | 5324.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 09:15:00 | 5329.05 | 5500.86 | 5324.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 5345.00 | 5499.31 | 5324.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 14:45:00 | 5391.45 | 5492.25 | 5325.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 5488.20 | 5491.11 | 5325.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 5456.30 | 5488.20 | 5340.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 5384.95 | 5483.28 | 5342.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 5342.10 | 5480.91 | 5342.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 5336.30 | 5480.91 | 5342.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 5385.00 | 5479.95 | 5342.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 5410.50 | 5479.95 | 5342.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 5391.90 | 5475.69 | 5344.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:30:00 | 5393.75 | 5474.86 | 5344.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 14:15:00 | 5392.80 | 5472.12 | 5345.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-28 09:15:00 | 5930.60 | 5567.10 | 5434.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 5608.70 | 6087.84 | 6087.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 5572.50 | 5855.16 | 5916.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 10:15:00 | 4530.00 | 4476.58 | 4822.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 11:00:00 | 4530.00 | 4476.58 | 4822.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 4778.50 | 4533.90 | 4737.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 4778.50 | 4533.90 | 4737.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 4780.50 | 4536.35 | 4738.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 4780.50 | 4536.35 | 4738.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 5131.40 | 4858.04 | 4857.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 5140.00 | 4860.85 | 4858.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 5233.00 | 5276.11 | 5164.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 5151.00 | 5271.05 | 5165.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 5151.00 | 5271.05 | 5165.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 5128.00 | 5271.05 | 5165.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 5136.00 | 5269.70 | 5165.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 5135.00 | 5269.70 | 5165.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 5128.00 | 5268.29 | 5165.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:45:00 | 5134.00 | 5268.29 | 5165.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 5164.00 | 5263.64 | 5164.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 5215.00 | 5263.64 | 5164.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 5112.50 | 5260.35 | 5173.49 | SL hit (close<static) qty=1.00 sl=5154.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 5107.50 | 5135.86 | 5135.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 5080.00 | 5134.69 | 5135.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 5172.00 | 5133.14 | 5134.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 5229.00 | 5134.09 | 5135.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 5229.00 | 5134.09 | 5135.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 5135.00 | 5134.15 | 5135.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 5135.00 | 5134.15 | 5135.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 5154.00 | 5134.35 | 5135.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 5166.50 | 5134.35 | 5135.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 5104.50 | 5134.05 | 5134.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 5098.50 | 5133.82 | 5134.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 5087.50 | 5133.12 | 5134.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 5098.50 | 5127.93 | 5131.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 5100.00 | 5127.70 | 5131.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 5133.50 | 5127.54 | 5131.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 5133.50 | 5127.54 | 5131.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 5212.00 | 5128.38 | 5131.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 5212.00 | 5128.38 | 5131.89 | SL hit (close>static) qty=1.00 sl=5155.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 5243.00 | 5135.77 | 5135.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 5315.50 | 5145.70 | 5140.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 5156.50 | 5160.42 | 5148.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:00:00 | 5156.50 | 5160.42 | 5148.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 5153.00 | 5160.34 | 5148.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 5092.00 | 5160.34 | 5148.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 5096.00 | 5159.70 | 5148.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 5096.00 | 5159.70 | 5148.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 5098.50 | 5159.09 | 5148.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 5071.50 | 5159.09 | 5148.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 5130.00 | 5157.78 | 5147.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 5129.00 | 5157.78 | 5147.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 5141.00 | 5157.61 | 5147.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:30:00 | 5103.50 | 5157.12 | 5147.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 5132.50 | 5156.87 | 5147.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 5140.00 | 5156.41 | 5147.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 5154.00 | 5155.64 | 5146.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 5183.00 | 5165.66 | 5153.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 5143.00 | 5166.69 | 5154.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 5154.50 | 5166.57 | 5154.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 5164.00 | 5166.57 | 5154.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 5137.00 | 5166.28 | 5154.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 5137.00 | 5166.28 | 5154.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 5079.00 | 5165.41 | 5154.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 5079.00 | 5165.41 | 5154.08 | SL hit (close<static) qty=1.00 sl=5103.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 09:15:00 | 5648.00 | 5926.75 | 5926.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 5584.00 | 5899.10 | 5912.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 15:15:00 | 4454.90 | 4453.33 | 4829.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 09:15:00 | 4475.70 | 4453.33 | 4829.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 4760.40 | 4497.26 | 4776.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:00:00 | 4760.40 | 4497.26 | 4776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 4759.40 | 4527.13 | 4773.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 4761.20 | 4527.13 | 4773.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 4762.50 | 4529.47 | 4773.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:45:00 | 4765.00 | 4529.47 | 4773.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 12:45:00 | 4867.40 | 2024-06-04 09:15:00 | 4624.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 4858.15 | 2024-06-04 09:15:00 | 4615.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 09:15:00 | 4832.15 | 2024-06-04 09:15:00 | 4590.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 12:45:00 | 4867.40 | 2024-06-06 12:15:00 | 4780.25 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2024-05-29 09:15:00 | 4858.15 | 2024-06-06 12:15:00 | 4780.25 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2024-05-30 09:15:00 | 4832.15 | 2024-06-06 12:15:00 | 4780.25 | STOP_HIT | 0.50 | 1.07% |
| BUY | retest2 | 2024-08-05 14:45:00 | 5391.45 | 2024-08-28 09:15:00 | 5930.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 5488.20 | 2024-08-28 09:15:00 | 5923.45 | TARGET_HIT | 1.00 | 7.93% |
| BUY | retest2 | 2024-08-09 09:15:00 | 5456.30 | 2024-08-28 09:15:00 | 5951.55 | TARGET_HIT | 1.00 | 9.08% |
| BUY | retest2 | 2024-08-09 15:15:00 | 5384.95 | 2024-08-28 09:15:00 | 5931.09 | TARGET_HIT | 1.00 | 10.14% |
| BUY | retest2 | 2024-08-12 11:15:00 | 5410.50 | 2024-08-28 09:15:00 | 5933.13 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2024-08-13 10:00:00 | 5391.90 | 2024-08-28 09:15:00 | 5932.08 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2024-08-13 10:30:00 | 5393.75 | 2024-08-28 10:15:00 | 6037.02 | TARGET_HIT | 1.00 | 11.93% |
| BUY | retest2 | 2024-08-13 14:15:00 | 5392.80 | 2024-08-28 10:15:00 | 6001.93 | TARGET_HIT | 1.00 | 11.30% |
| BUY | retest2 | 2024-11-22 13:45:00 | 6088.00 | 2024-12-12 09:15:00 | 6696.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-20 10:00:00 | 6098.30 | 2024-12-20 11:15:00 | 5977.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-07-15 09:15:00 | 5215.00 | 2025-07-18 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-22 09:15:00 | 5174.00 | 2025-07-25 09:15:00 | 5126.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-22 10:00:00 | 5170.00 | 2025-07-25 09:15:00 | 5126.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-18 09:15:00 | 5098.50 | 2025-08-20 10:15:00 | 5212.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-18 10:45:00 | 5087.50 | 2025-08-20 10:15:00 | 5212.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-08-19 13:15:00 | 5098.50 | 2025-08-20 10:15:00 | 5212.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-19 13:45:00 | 5100.00 | 2025-08-20 10:15:00 | 5212.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-08-29 13:15:00 | 5140.00 | 2025-09-08 14:15:00 | 5079.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-09-01 09:15:00 | 5154.00 | 2025-09-08 14:15:00 | 5079.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-09-05 10:15:00 | 5183.00 | 2025-09-08 14:15:00 | 5079.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-08 12:00:00 | 5143.00 | 2025-09-08 14:15:00 | 5079.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-08 09:30:00 | 5391.00 | 2025-11-19 10:15:00 | 5930.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-09 12:15:00 | 5389.50 | 2025-11-19 10:15:00 | 5928.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-09 13:45:00 | 5385.50 | 2025-11-19 10:15:00 | 5924.05 | TARGET_HIT | 1.00 | 10.00% |
