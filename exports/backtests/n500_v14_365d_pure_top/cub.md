# City Union Bank Ltd. (CUB)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 258.95
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 0
- **Avg / median % per leg:** 0.71% / -1.87%
- **Sum % (uncompounded):** 9.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.15% | 13.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.15% | 13.8% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.92% | -3.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.92% | -3.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 4 | 28.6% | 4 | 10 | 0 | 0.71% | 9.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 246.70 | 275.28 | 275.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 246.30 | 275.00 | 275.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 257.50 | 252.87 | 260.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 14:15:00 | 255.00 | 252.98 | 260.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 255.00 | 252.98 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:45:00 | 256.30 | 252.98 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 259.51 | 252.97 | 259.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 263.48 | 252.97 | 259.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 265.98 | 253.10 | 259.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 266.00 | 253.10 | 259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 269.89 | 253.27 | 259.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:00:00 | 269.89 | 253.27 | 259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 261.03 | 256.56 | 260.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 260.46 | 256.60 | 260.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 260.75 | 256.60 | 260.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 265.62 | 256.81 | 260.32 | SL hit (close>static) qty=1.00 sl=262.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 265.62 | 256.81 | 260.32 | SL hit (close>static) qty=1.00 sl=262.68 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 269.10 | 263.15 | 263.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 269.85 | 263.28 | 263.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 262.00 | 263.44 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 261.80 | 263.43 | 263.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 261.75 | 263.43 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-31 11:45:00 | 210.60 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2025-08-11 15:00:00 | 209.27 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-08-12 12:30:00 | 209.65 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2025-08-12 13:00:00 | 209.34 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-09-19 12:30:00 | 209.00 | 2025-09-22 14:15:00 | 206.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-22 12:15:00 | 209.43 | 2025-09-22 14:15:00 | 206.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-24 09:30:00 | 208.88 | 2025-10-20 12:15:00 | 229.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 10:15:00 | 208.64 | 2025-10-20 12:15:00 | 229.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-13 09:15:00 | 269.75 | 2026-01-28 15:15:00 | 294.03 | TARGET_HIT | 1.00 | 9.00% |
| BUY | retest2 | 2026-01-20 10:00:00 | 267.30 | 2026-01-30 09:15:00 | 296.73 | TARGET_HIT | 1.00 | 11.01% |
| BUY | retest2 | 2026-03-05 09:15:00 | 267.75 | 2026-03-06 14:15:00 | 260.70 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-03-05 10:00:00 | 266.50 | 2026-03-06 14:15:00 | 260.70 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-24 13:30:00 | 260.46 | 2026-04-27 09:15:00 | 265.62 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-04-24 14:00:00 | 260.75 | 2026-04-27 09:15:00 | 265.62 | STOP_HIT | 1.00 | -1.87% |
