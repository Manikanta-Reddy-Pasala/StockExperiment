# India Cements Ltd. (INDIACEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 408.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 21
- **Target hits / Stop hits / Partials:** 3 / 21 / 0
- **Avg / median % per leg:** -0.93% / -1.37%
- **Sum % (uncompounded):** -22.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 3 | 9 | 0 | 0.75% | 9.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 3 | 9 | 0 | 0.75% | 9.0% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.62% | -31.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.62% | -31.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 3 | 12.5% | 3 | 21 | 0 | -0.93% | -22.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 263.78 | 219.39 | 219.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 264.80 | 219.84 | 219.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 361.90 | 362.19 | 342.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 15:00:00 | 361.90 | 362.19 | 342.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 352.10 | 361.57 | 353.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 12:45:00 | 357.80 | 360.97 | 353.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:45:00 | 357.40 | 360.60 | 353.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 12:00:00 | 355.85 | 360.64 | 354.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 12:30:00 | 356.65 | 360.59 | 354.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 355.30 | 360.14 | 355.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 355.30 | 360.14 | 355.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 355.45 | 360.10 | 355.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 357.35 | 359.24 | 355.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 14:30:00 | 356.65 | 359.03 | 355.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 11:15:00 | 355.05 | 358.89 | 355.18 | SL hit (close<static) qty=1.00 sl=355.10 alert=retest2 |

### Cycle 2 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 332.60 | 354.85 | 354.91 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 376.45 | 354.54 | 354.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 378.90 | 356.09 | 355.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 368.25 | 370.43 | 364.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 368.25 | 370.43 | 364.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 355.85 | 370.28 | 364.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 355.85 | 370.28 | 364.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 359.40 | 370.17 | 364.67 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 297.70 | 359.71 | 359.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 276.10 | 357.00 | 358.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 12:15:00 | 280.90 | 279.98 | 303.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 12:30:00 | 280.65 | 279.98 | 303.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 287.95 | 278.27 | 288.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 285.75 | 278.27 | 288.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 294.40 | 278.57 | 288.08 | SL hit (close>static) qty=1.00 sl=289.25 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 13:15:00 | 315.10 | 292.78 | 292.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 319.00 | 294.97 | 293.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 312.65 | 315.32 | 306.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 09:15:00 | 317.60 | 315.32 | 306.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 319.00 | 327.17 | 317.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 319.00 | 327.17 | 317.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 319.80 | 327.10 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 317.80 | 327.10 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 316.30 | 326.83 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 315.50 | 326.83 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 317.60 | 326.74 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 316.65 | 326.74 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 315.05 | 326.62 | 317.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 315.05 | 326.62 | 317.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 312.85 | 326.48 | 317.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 312.95 | 326.48 | 317.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 316.60 | 324.58 | 317.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 316.60 | 324.58 | 317.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 315.65 | 324.49 | 317.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 315.65 | 324.49 | 317.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 312.35 | 324.37 | 317.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 312.35 | 324.37 | 317.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 337.20 | 339.25 | 330.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 343.00 | 339.26 | 330.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:30:00 | 342.85 | 339.28 | 330.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 358.85 | 339.31 | 330.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 377.30 | 356.96 | 345.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 387.10 | 432.68 | 432.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 382.55 | 426.25 | 429.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 376.75 | 375.97 | 394.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 376.75 | 375.97 | 394.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 393.20 | 377.82 | 392.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 393.20 | 377.82 | 392.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 395.90 | 378.00 | 392.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:15:00 | 396.00 | 378.00 | 392.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 392.40 | 398.32 | 400.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 391.15 | 398.32 | 400.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:45:00 | 391.70 | 398.26 | 400.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 391.10 | 397.96 | 400.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 391.75 | 397.69 | 399.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 392.55 | 397.46 | 399.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 411.10 | 397.64 | 399.74 | SL hit (close>static) qty=1.00 sl=404.85 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-28 12:45:00 | 357.80 | 2024-11-21 11:15:00 | 355.05 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-10-30 09:45:00 | 357.40 | 2024-11-21 11:15:00 | 355.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-11-08 12:00:00 | 355.85 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-11-08 12:30:00 | 356.65 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-18 14:00:00 | 357.35 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-11-19 14:30:00 | 356.65 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-22 11:45:00 | 356.65 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-25 09:15:00 | 358.95 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-12-11 09:15:00 | 349.50 | 2024-12-11 10:15:00 | 344.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-04-16 13:15:00 | 285.75 | 2025-04-17 09:15:00 | 294.40 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-04-21 09:15:00 | 285.90 | 2025-04-22 15:15:00 | 289.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-04-21 11:15:00 | 286.70 | 2025-04-22 15:15:00 | 289.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-04-22 11:15:00 | 285.00 | 2025-04-22 15:15:00 | 289.50 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-04-23 09:15:00 | 287.65 | 2025-04-24 09:15:00 | 291.15 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-04-23 14:00:00 | 288.05 | 2025-04-24 09:15:00 | 291.15 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-04-23 15:15:00 | 288.00 | 2025-04-24 09:15:00 | 291.15 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-04-25 09:45:00 | 287.00 | 2025-04-28 12:15:00 | 290.05 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-21 14:00:00 | 343.00 | 2025-08-18 09:15:00 | 377.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 14:30:00 | 342.85 | 2025-08-18 09:15:00 | 377.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 09:15:00 | 358.85 | 2025-08-21 12:15:00 | 394.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 391.15 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2026-04-30 10:45:00 | 391.70 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-05-04 12:00:00 | 391.10 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -5.11% |
| SELL | retest2 | 2026-05-05 09:15:00 | 391.75 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -4.94% |
