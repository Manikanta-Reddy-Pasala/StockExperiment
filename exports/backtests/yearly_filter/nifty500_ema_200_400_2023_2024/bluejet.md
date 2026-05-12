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
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 32
- **Target hits / Stop hits / Partials:** 3 / 33 / 1
- **Avg / median % per leg:** -1.51% / -2.21%
- **Sum % (uncompounded):** -55.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 3 | 11.1% | 3 | 24 | 0 | -0.75% | -20.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 3 | 11.1% | 3 | 24 | 0 | -0.75% | -20.2% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -3.57% | -35.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -3.57% | -35.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 5 | 13.5% | 3 | 33 | 1 | -1.51% | -56.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 14:15:00 | 357.70 | 372.07 | 372.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 356.15 | 371.33 | 371.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 13:15:00 | 372.25 | 370.74 | 371.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 13:15:00 | 372.25 | 370.74 | 371.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 13:15:00 | 372.25 | 370.74 | 371.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 13:45:00 | 372.45 | 370.74 | 371.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 382.10 | 370.85 | 371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 15:00:00 | 382.10 | 370.85 | 371.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 362.95 | 353.60 | 360.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:45:00 | 364.00 | 353.60 | 360.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 358.05 | 353.64 | 360.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 11:15:00 | 355.60 | 353.64 | 360.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 356.95 | 353.10 | 359.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 365.85 | 354.11 | 359.58 | SL hit (close>static) qty=1.00 sl=364.70 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 14:15:00 | 393.30 | 358.48 | 358.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 09:15:00 | 395.60 | 362.72 | 360.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 12:15:00 | 379.10 | 379.67 | 371.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 12:45:00 | 379.00 | 379.67 | 371.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 378.05 | 381.88 | 374.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:45:00 | 377.00 | 381.88 | 374.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 373.90 | 381.80 | 374.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:00:00 | 373.90 | 381.80 | 374.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 374.00 | 381.72 | 374.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:00:00 | 374.00 | 381.72 | 374.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 370.00 | 381.60 | 374.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:45:00 | 369.80 | 381.60 | 374.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 369.90 | 381.49 | 374.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 369.90 | 381.49 | 374.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 371.95 | 381.16 | 374.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:15:00 | 372.70 | 381.16 | 374.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 12:00:00 | 372.70 | 381.08 | 374.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 11:30:00 | 372.85 | 380.89 | 374.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 12:15:00 | 373.05 | 380.89 | 374.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 375.00 | 380.83 | 374.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 14:45:00 | 380.00 | 380.77 | 374.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 376.80 | 380.70 | 374.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 377.00 | 380.65 | 374.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 11:15:00 | 379.45 | 380.65 | 374.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 379.60 | 380.63 | 374.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 371.95 | 380.63 | 374.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 368.65 | 380.42 | 375.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 368.65 | 380.42 | 375.05 | SL hit (close<static) qty=1.00 sl=369.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-06 13:15:00)

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

### Cycle 4 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 482.75 | 417.93 | 417.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 487.95 | 418.63 | 418.01 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-02-21 11:15:00 | 355.60 | 2024-02-27 13:15:00 | 365.85 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-02-26 09:15:00 | 356.95 | 2024-02-27 13:15:00 | 365.85 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-02-28 09:15:00 | 356.25 | 2024-03-06 10:15:00 | 338.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-28 09:15:00 | 356.25 | 2024-03-07 11:15:00 | 354.00 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2024-03-20 10:00:00 | 357.00 | 2024-03-21 14:15:00 | 364.95 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-05-06 11:15:00 | 372.70 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-05-06 12:00:00 | 372.70 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-05-07 11:30:00 | 372.85 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-05-07 12:15:00 | 373.05 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-07 14:45:00 | 380.00 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-05-09 10:00:00 | 376.80 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-05-09 10:30:00 | 377.00 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-05-09 11:15:00 | 379.45 | 2024-05-13 09:15:00 | 368.65 | STOP_HIT | 1.00 | -2.85% |
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
