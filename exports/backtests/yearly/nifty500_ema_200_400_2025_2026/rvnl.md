# Rail Vikas Nigam Ltd. (RVNL)

## Backtest Summary

- **Window:** 2025-01-16 09:15:00 → 2026-05-08 15:15:00 (2256 bars)
- **Last close:** 305.15
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
| ALERT2_SKIP | 0 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 4
- **Avg / median % per leg:** 1.17% / -1.28%
- **Sum % (uncompounded):** 28.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |
| SELL (all) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.48% | 53.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.48% | 53.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 8 | 33.3% | 4 | 16 | 4 | 1.17% | 28.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 412.00 | 370.61 | 370.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 413.20 | 378.16 | 374.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 400.40 | 403.79 | 391.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 398.95 | 403.79 | 391.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 392.90 | 403.53 | 392.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 392.40 | 403.53 | 392.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 387.20 | 403.37 | 392.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 387.20 | 403.37 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 381.60 | 403.15 | 392.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 381.60 | 403.15 | 392.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 390.60 | 401.45 | 392.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 393.90 | 401.28 | 392.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:45:00 | 393.10 | 401.02 | 392.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 400.10 | 400.92 | 392.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 392.80 | 399.85 | 393.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 393.95 | 399.43 | 393.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 393.60 | 399.43 | 393.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 392.85 | 399.37 | 393.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 392.80 | 399.37 | 393.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 392.60 | 399.30 | 393.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 392.35 | 399.30 | 393.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 392.00 | 398.92 | 393.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 391.90 | 398.92 | 393.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 389.95 | 398.75 | 393.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 389.25 | 398.75 | 393.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 391.20 | 398.52 | 393.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 393.85 | 398.52 | 393.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 392.20 | 398.45 | 393.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 389.40 | 398.29 | 393.37 | SL hit (close<static) qty=1.00 sl=390.25 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 378.60 | 390.13 | 390.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 375.75 | 389.87 | 390.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 338.35 | 338.27 | 355.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 338.35 | 338.27 | 355.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 352.80 | 336.12 | 349.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 354.10 | 336.12 | 349.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 352.50 | 336.29 | 349.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:15:00 | 353.50 | 336.29 | 349.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 350.35 | 337.04 | 349.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 349.45 | 337.04 | 349.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 348.85 | 337.16 | 349.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 347.65 | 337.39 | 349.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 354.90 | 337.69 | 349.53 | SL hit (close>static) qty=1.00 sl=350.40 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 362.30 | 329.99 | 329.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 365.45 | 335.22 | 332.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 10:15:00 | 340.75 | 341.17 | 336.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:00:00 | 340.75 | 341.17 | 336.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 336.25 | 341.12 | 336.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 336.25 | 341.12 | 336.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 334.65 | 341.06 | 336.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 334.65 | 341.06 | 336.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 332.20 | 340.97 | 336.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 332.20 | 340.97 | 336.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 332.55 | 340.89 | 336.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 333.20 | 340.89 | 336.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 330.75 | 339.81 | 335.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 330.80 | 339.81 | 335.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 337.80 | 339.10 | 335.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 339.20 | 339.09 | 335.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 338.55 | 339.09 | 335.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:00:00 | 339.50 | 339.09 | 335.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 334.20 | 339.02 | 335.83 | SL hit (close<static) qty=1.00 sl=335.25 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 323.70 | 333.50 | 333.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 321.35 | 332.77 | 333.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 321.60 | 320.02 | 325.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 321.60 | 320.02 | 325.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 288.30 | 276.67 | 290.42 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 12:00:00 | 393.90 | 2025-07-07 11:15:00 | 389.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-06-23 14:45:00 | 393.10 | 2025-07-07 11:15:00 | 389.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-06-24 09:15:00 | 400.10 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-07-02 11:45:00 | 392.80 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-07 09:15:00 | 393.85 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-07-07 10:15:00 | 392.20 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-09-16 13:15:00 | 347.65 | 2025-09-16 14:15:00 | 354.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-09-24 14:45:00 | 347.50 | 2025-09-25 09:15:00 | 353.30 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-25 13:15:00 | 348.30 | 2025-10-07 12:15:00 | 351.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-25 15:00:00 | 346.20 | 2025-10-07 12:15:00 | 351.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-08 13:30:00 | 346.70 | 2025-10-17 13:15:00 | 329.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 14:00:00 | 346.55 | 2025-10-17 13:15:00 | 329.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 09:15:00 | 343.00 | 2025-10-17 13:15:00 | 329.36 | PARTIAL | 0.50 | 3.98% |
| SELL | retest2 | 2025-10-10 09:30:00 | 346.70 | 2025-11-04 13:15:00 | 325.85 | PARTIAL | 0.50 | 6.01% |
| SELL | retest2 | 2025-10-08 13:30:00 | 346.70 | 2025-11-12 09:15:00 | 312.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-08 14:00:00 | 346.55 | 2025-11-12 09:15:00 | 311.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-09 09:15:00 | 343.00 | 2025-11-12 09:15:00 | 308.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-10 09:30:00 | 346.70 | 2025-11-12 09:15:00 | 312.03 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-14 14:30:00 | 339.20 | 2026-01-16 13:15:00 | 334.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-14 15:15:00 | 338.55 | 2026-01-16 13:15:00 | 334.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-16 12:00:00 | 339.50 | 2026-01-16 13:15:00 | 334.20 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-01-28 13:00:00 | 338.45 | 2026-01-29 10:15:00 | 332.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-29 14:30:00 | 336.90 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-30 10:00:00 | 338.95 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -2.66% |
