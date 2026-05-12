# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 0
- **Avg / median % per leg:** 0.81% / -1.24%
- **Sum % (uncompounded):** 11.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 3 | 7 | 0 | 2.04% | 20.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 3 | 7 | 0 | 2.04% | 20.4% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.27% | -9.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.27% | -9.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 3 | 21.4% | 3 | 11 | 0 | 0.81% | 11.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 5131.40 | 4858.04 | 4857.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 5140.00 | 4860.85 | 4858.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 5233.00 | 5276.11 | 5164.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 5151.00 | 5271.05 | 5165.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 5151.00 | 5271.05 | 5165.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 5128.00 | 5271.05 | 5165.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 5136.00 | 5269.70 | 5165.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 5135.00 | 5269.70 | 5165.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 5128.00 | 5268.29 | 5165.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:45:00 | 5134.00 | 5268.29 | 5165.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 5164.00 | 5263.64 | 5164.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 5215.00 | 5263.64 | 5164.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 5112.50 | 5260.35 | 5173.47 | SL hit (close<static) qty=1.00 sl=5154.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 5107.50 | 5135.86 | 5135.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 5080.00 | 5134.69 | 5135.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 5172.00 | 5133.14 | 5134.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 5229.00 | 5134.09 | 5135.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 5229.00 | 5134.09 | 5135.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 5135.00 | 5134.15 | 5135.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 5135.00 | 5134.15 | 5135.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 5154.00 | 5134.35 | 5135.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 5166.50 | 5134.35 | 5135.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 5104.50 | 5134.05 | 5134.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 5098.50 | 5133.82 | 5134.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 5087.50 | 5133.12 | 5134.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 5098.50 | 5127.93 | 5131.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 5100.00 | 5127.70 | 5131.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 5133.50 | 5127.54 | 5131.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 5133.50 | 5127.54 | 5131.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 5212.00 | 5128.38 | 5131.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 5212.00 | 5128.38 | 5131.89 | SL hit (close>static) qty=1.00 sl=5155.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 5243.00 | 5135.77 | 5135.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 5315.50 | 5145.70 | 5140.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 5156.50 | 5160.42 | 5148.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:00:00 | 5156.50 | 5160.42 | 5148.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 5153.00 | 5160.34 | 5148.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 5092.00 | 5160.34 | 5148.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 5096.00 | 5159.70 | 5148.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 5096.00 | 5159.70 | 5148.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 5098.50 | 5159.09 | 5148.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 5071.50 | 5159.09 | 5148.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 5130.00 | 5157.78 | 5147.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 5129.00 | 5157.78 | 5147.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 5141.00 | 5157.61 | 5147.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:30:00 | 5103.50 | 5157.12 | 5147.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 5132.50 | 5156.87 | 5147.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 5140.00 | 5156.41 | 5147.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 5154.00 | 5155.64 | 5146.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 5183.00 | 5165.66 | 5153.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 5143.00 | 5166.69 | 5154.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 5154.50 | 5166.57 | 5154.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 5164.00 | 5166.57 | 5154.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 5137.00 | 5166.28 | 5154.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 5137.00 | 5166.28 | 5154.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 5079.00 | 5165.41 | 5154.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 5079.00 | 5165.41 | 5154.08 | SL hit (close<static) qty=1.00 sl=5103.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-10 09:15:00)

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
