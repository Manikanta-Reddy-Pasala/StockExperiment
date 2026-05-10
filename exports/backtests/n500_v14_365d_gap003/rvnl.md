# Rail Vikas Nigam Ltd. (RVNL)

## Backtest Summary

- **Window:** 2024-10-14 09:15:00 → 2026-05-08 15:15:00 (2706 bars)
- **Last close:** 305.15
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
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -2.14% / -1.64%
- **Sum % (uncompounded):** -25.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 414.75 | 368.75 | 368.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 417.65 | 380.17 | 374.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 400.40 | 403.74 | 391.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 398.95 | 403.74 | 391.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 392.90 | 403.49 | 392.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 392.40 | 403.49 | 392.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 387.20 | 403.33 | 392.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 387.20 | 403.33 | 392.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 381.60 | 403.11 | 392.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 381.60 | 403.11 | 392.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 390.60 | 401.41 | 391.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 393.90 | 401.24 | 391.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:45:00 | 393.10 | 400.98 | 391.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 400.10 | 400.88 | 391.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 392.80 | 399.83 | 393.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 392.80 | 399.53 | 393.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 389.35 | 399.53 | 393.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 393.10 | 399.47 | 393.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 391.40 | 399.47 | 393.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 393.95 | 399.41 | 393.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 393.60 | 399.41 | 393.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 392.85 | 399.35 | 393.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 392.80 | 399.35 | 393.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 392.60 | 399.28 | 393.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 392.35 | 399.28 | 393.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 392.00 | 398.90 | 393.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 391.90 | 398.90 | 393.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 389.95 | 398.73 | 393.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 389.25 | 398.73 | 393.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 391.20 | 398.50 | 393.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 393.85 | 398.50 | 393.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 392.20 | 398.44 | 393.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 389.40 | 398.28 | 393.01 | SL hit (close<static) qty=1.00 sl=390.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 389.40 | 398.28 | 393.01 | SL hit (close<static) qty=1.00 sl=390.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 382.00 | 395.56 | 392.21 | SL hit (close<static) qty=1.00 sl=384.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 382.00 | 395.56 | 392.21 | SL hit (close<static) qty=1.00 sl=384.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 382.00 | 395.56 | 392.21 | SL hit (close<static) qty=1.00 sl=384.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 382.00 | 395.56 | 392.21 | SL hit (close<static) qty=1.00 sl=384.45 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-22 11:15:00 | 374.80 | 389.56 | 389.63 | min_gap filter: gap=0.017% < 0.030% |
| TREND_RESET | 2025-07-22 11:15:00 | 374.80 | 389.56 | 389.63 | EMA inversion without crossover edge (EMA200=389.56 EMA400=389.63) — end cycle |

### Cycle 2 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 362.30 | 329.99 | 329.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 365.45 | 335.22 | 332.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 10:15:00 | 340.75 | 341.17 | 336.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:00:00 | 340.75 | 341.17 | 336.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 336.25 | 341.12 | 336.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 336.25 | 341.12 | 336.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 334.65 | 341.06 | 336.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 334.65 | 341.06 | 336.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 332.20 | 340.97 | 336.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 332.20 | 340.97 | 336.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 332.55 | 340.89 | 336.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 333.20 | 340.89 | 336.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 330.75 | 339.81 | 335.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 330.80 | 339.81 | 335.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 337.80 | 339.10 | 335.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 339.20 | 339.09 | 335.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 338.55 | 339.09 | 335.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:00:00 | 339.50 | 339.09 | 335.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 334.20 | 339.02 | 335.82 | SL hit (close<static) qty=1.00 sl=335.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 334.20 | 339.02 | 335.82 | SL hit (close<static) qty=1.00 sl=335.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 334.20 | 339.02 | 335.82 | SL hit (close<static) qty=1.00 sl=335.25 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 13:00:00 | 338.45 | 334.01 | 333.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 332.90 | 334.26 | 333.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 332.90 | 334.26 | 333.83 | SL hit (close<static) qty=1.00 sl=335.25 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 335.90 | 334.26 | 333.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 333.30 | 334.25 | 333.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 331.75 | 334.25 | 333.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 334.20 | 334.25 | 333.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 336.90 | 334.33 | 333.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 338.95 | 334.44 | 333.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 329.95 | 334.99 | 334.24 | SL hit (close<static) qty=1.00 sl=332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 329.95 | 334.99 | 334.24 | SL hit (close<static) qty=1.00 sl=332.50 alert=retest2 |
| CROSSOVER_SKIP | 2026-02-03 09:15:00 | 323.70 | 333.50 | 333.52 | min_gap filter: gap=0.006% < 0.030% |
| TREND_RESET | 2026-02-03 09:15:00 | 323.70 | 333.50 | 333.52 | EMA inversion without crossover edge (EMA200=333.50 EMA400=333.52) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 12:00:00 | 393.90 | 2025-07-07 11:15:00 | 389.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-06-23 14:45:00 | 393.10 | 2025-07-07 11:15:00 | 389.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-06-24 09:15:00 | 400.10 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-07-02 11:45:00 | 392.80 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-07 09:15:00 | 393.85 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-07-07 10:15:00 | 392.20 | 2025-07-11 09:15:00 | 382.00 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-01-14 14:30:00 | 339.20 | 2026-01-16 13:15:00 | 334.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-14 15:15:00 | 338.55 | 2026-01-16 13:15:00 | 334.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-16 12:00:00 | 339.50 | 2026-01-16 13:15:00 | 334.20 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-01-28 13:00:00 | 338.45 | 2026-01-29 10:15:00 | 332.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-29 14:30:00 | 336.90 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-30 10:00:00 | 338.95 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -2.66% |
