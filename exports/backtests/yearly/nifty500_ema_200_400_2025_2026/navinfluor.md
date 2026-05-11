# Navin Fluorine International Ltd. (NAVINFLUOR)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 7039.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 9 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 0
- **Target hits / Stop hits / Partials:** 9 / 0 / 0
- **Avg / median % per leg:** 10.00% / 10.00%
- **Sum % (uncompounded):** 90.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 9 | 100.0% | 9 | 0 | 0 | 10.00% | 90.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 4644.00 | 4807.09 | 4807.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 4617.60 | 4803.51 | 4805.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 10:15:00 | 4726.00 | 4710.56 | 4751.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 11:00:00 | 4726.00 | 4710.56 | 4751.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 4740.50 | 4710.83 | 4750.43 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 5110.00 | 4783.00 | 4782.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 5151.60 | 4797.71 | 4789.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 5642.50 | 5695.26 | 5451.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 5642.50 | 5695.26 | 5451.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 5568.50 | 5687.09 | 5456.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:30:00 | 5505.00 | 5687.09 | 5456.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 5792.50 | 5845.07 | 5693.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 5724.50 | 5845.07 | 5693.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 5706.50 | 5842.08 | 5696.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 5706.50 | 5842.08 | 5696.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5761.00 | 5896.52 | 5754.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 5762.50 | 5896.52 | 5754.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 5688.50 | 5894.45 | 5754.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 5645.00 | 5894.45 | 5754.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 5712.00 | 5892.64 | 5753.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:30:00 | 5615.00 | 5892.64 | 5753.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5725.00 | 5886.68 | 5754.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 5725.00 | 5886.68 | 5754.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5722.00 | 5885.05 | 5754.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 5739.00 | 5885.05 | 5754.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 5739.50 | 5883.60 | 5753.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:45:00 | 5711.50 | 5883.60 | 5753.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 5755.00 | 5885.82 | 5763.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 5755.00 | 5885.82 | 5763.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 5803.00 | 5885.00 | 5763.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 5829.50 | 5882.15 | 5763.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 6412.45 | 5925.86 | 5807.62 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-27 14:30:00 | 5829.50 | 2026-02-03 09:15:00 | 6412.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:30:00 | 5822.00 | 2026-04-15 09:15:00 | 6404.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:15:00 | 5820.50 | 2026-04-15 09:15:00 | 6402.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:00:00 | 5825.00 | 2026-04-15 09:15:00 | 6407.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 12:30:00 | 6191.00 | 2026-04-29 09:15:00 | 6810.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 09:15:00 | 6149.00 | 2026-04-29 09:15:00 | 6763.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 10:00:00 | 6166.00 | 2026-04-29 09:15:00 | 6782.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 12:45:00 | 6157.50 | 2026-04-29 09:15:00 | 6773.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:30:00 | 6187.00 | 2026-04-29 09:15:00 | 6805.70 | TARGET_HIT | 1.00 | 10.00% |
