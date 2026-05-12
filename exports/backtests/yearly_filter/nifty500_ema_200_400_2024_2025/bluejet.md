# Blue Jet Healthcare Ltd. (BLUEJET)

## Backtest Summary

- **Window:** 2023-11-01 09:15:00 → 2026-05-11 15:15:00 (4351 bars)
- **Last close:** 476.00
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 21
- **Target hits / Stop hits / Partials:** 3 / 21 / 0
- **Avg / median % per leg:** -1.64% / -2.29%
- **Sum % (uncompounded):** -39.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 3 | 15.8% | 3 | 16 | 0 | -0.29% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 3 | 15.8% | 3 | 16 | 0 | -0.29% | -5.6% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -6.75% | -33.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -6.75% | -33.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 3 | 12.5% | 3 | 21 | 0 | -1.64% | -39.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 766.80 | 852.26 | 852.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 760.90 | 851.35 | 851.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 674.15 | 666.61 | 699.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:00:00 | 674.15 | 666.61 | 699.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 406.85 | 371.27 | 403.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:30:00 | 406.85 | 371.27 | 403.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 406.85 | 371.62 | 403.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:30:00 | 406.85 | 371.62 | 403.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 401.60 | 373.55 | 403.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:45:00 | 402.20 | 373.55 | 403.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 404.15 | 374.65 | 403.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:30:00 | 403.90 | 374.65 | 403.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 403.00 | 374.93 | 403.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 410.90 | 374.93 | 403.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 410.80 | 375.29 | 403.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 408.25 | 375.97 | 403.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:45:00 | 407.75 | 376.97 | 403.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 407.55 | 377.62 | 403.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 428.50 | 380.18 | 404.36 | SL hit (close>static) qty=1.00 sl=419.95 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 482.75 | 417.93 | 417.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 487.95 | 418.63 | 418.01 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 11:30:00 | 372.25 | 2024-05-15 11:15:00 | 368.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-05-14 12:45:00 | 372.95 | 2024-05-15 11:15:00 | 368.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-15 09:15:00 | 372.95 | 2024-05-15 11:15:00 | 368.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-21 09:45:00 | 372.20 | 2024-05-21 10:15:00 | 368.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-05-22 10:45:00 | 379.70 | 2024-05-28 11:15:00 | 371.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-05-22 14:45:00 | 379.50 | 2024-05-28 12:15:00 | 369.05 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-05-23 09:15:00 | 379.85 | 2024-05-28 12:15:00 | 369.05 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-05-23 14:15:00 | 378.55 | 2024-05-28 12:15:00 | 369.05 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-05-24 13:30:00 | 374.25 | 2024-05-28 12:15:00 | 369.05 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-05-27 10:00:00 | 374.10 | 2024-05-29 09:15:00 | 368.75 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-05-27 12:00:00 | 374.20 | 2024-05-29 09:15:00 | 368.75 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-05-27 12:30:00 | 376.80 | 2024-05-29 09:15:00 | 368.75 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-05-27 14:15:00 | 378.00 | 2024-05-29 09:15:00 | 368.75 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-05-30 10:15:00 | 377.05 | 2024-06-04 10:15:00 | 362.40 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-05-30 15:15:00 | 376.50 | 2024-06-04 10:15:00 | 362.40 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2024-06-04 09:45:00 | 377.45 | 2024-06-04 10:15:00 | 362.40 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-06-06 09:15:00 | 371.35 | 2024-06-19 12:15:00 | 408.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 12:30:00 | 370.35 | 2024-06-19 12:15:00 | 407.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 14:15:00 | 372.45 | 2024-06-19 12:15:00 | 409.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-15 11:45:00 | 408.25 | 2026-04-17 09:15:00 | 428.50 | STOP_HIT | 1.00 | -4.96% |
| SELL | retest2 | 2026-04-15 14:45:00 | 407.75 | 2026-04-17 09:15:00 | 428.50 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2026-04-16 09:45:00 | 407.55 | 2026-04-17 09:15:00 | 428.50 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2026-04-24 09:30:00 | 408.00 | 2026-04-27 09:15:00 | 443.65 | STOP_HIT | 1.00 | -8.74% |
| SELL | retest2 | 2026-04-24 11:45:00 | 404.00 | 2026-04-27 09:15:00 | 443.65 | STOP_HIT | 1.00 | -9.81% |
