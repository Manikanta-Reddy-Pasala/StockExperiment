# Rail Vikas Nigam Ltd. (RVNL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 305.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 10 |
| TARGET_HIT | 7 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 22
- **Target hits / Stop hits / Partials:** 7 / 25 / 10
- **Avg / median % per leg:** 2.05% / -0.92%
- **Sum % (uncompounded):** 86.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.14% | -25.6% |
| SELL (all) | 30 | 20 | 66.7% | 7 | 13 | 10 | 3.72% | 111.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 20 | 66.7% | 7 | 13 | 10 | 3.72% | 111.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 20 | 47.6% | 7 | 25 | 10 | 2.05% | 86.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 481.50 | 522.89 | 523.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 468.60 | 519.59 | 521.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 478.65 | 475.10 | 492.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:30:00 | 481.45 | 475.10 | 492.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 465.65 | 447.74 | 465.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 465.65 | 447.74 | 465.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 464.50 | 447.91 | 465.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 469.25 | 447.91 | 465.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 468.15 | 448.11 | 465.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:45:00 | 470.50 | 448.11 | 465.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 468.90 | 448.32 | 465.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 467.15 | 448.51 | 465.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 467.35 | 448.88 | 465.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 459.80 | 449.32 | 465.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 474.30 | 450.21 | 465.11 | SL hit (close>static) qty=1.00 sl=471.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 409.85 | 369.16 | 369.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 416.15 | 379.44 | 374.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 400.40 | 403.74 | 391.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 398.95 | 403.74 | 391.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 392.90 | 403.49 | 392.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 392.40 | 403.49 | 392.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 387.20 | 403.33 | 392.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 387.20 | 403.33 | 392.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 381.60 | 403.11 | 392.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 381.60 | 403.11 | 392.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 390.60 | 401.41 | 392.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 393.90 | 401.24 | 392.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:45:00 | 393.10 | 400.98 | 392.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 400.10 | 400.88 | 392.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 392.80 | 399.83 | 393.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 392.80 | 399.53 | 393.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 389.35 | 399.53 | 393.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 393.10 | 399.47 | 393.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 391.40 | 399.47 | 393.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 393.95 | 399.41 | 393.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 393.60 | 399.41 | 393.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 392.85 | 399.35 | 393.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 392.80 | 399.35 | 393.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 392.60 | 399.28 | 393.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 392.35 | 399.28 | 393.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 392.00 | 398.91 | 393.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 391.90 | 398.91 | 393.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 389.95 | 398.73 | 393.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 389.25 | 398.73 | 393.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 391.20 | 398.50 | 393.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 393.85 | 398.50 | 393.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 392.20 | 398.44 | 393.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 389.40 | 398.28 | 393.10 | SL hit (close<static) qty=1.00 sl=390.25 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 375.10 | 389.71 | 389.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 374.50 | 389.41 | 389.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 338.35 | 338.27 | 355.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 338.35 | 338.27 | 355.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 352.80 | 336.12 | 349.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 354.10 | 336.12 | 349.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 352.50 | 336.29 | 349.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:15:00 | 353.50 | 336.29 | 349.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 350.35 | 337.04 | 349.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 349.45 | 337.04 | 349.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 348.85 | 337.16 | 349.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 347.65 | 337.39 | 349.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 354.90 | 337.69 | 349.48 | SL hit (close>static) qty=1.00 sl=350.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-30 13:15:00)

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
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 334.20 | 339.02 | 335.83 | SL hit (close<static) qty=1.00 sl=335.25 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-03 09:15:00)

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
| SELL | retest2 | 2024-12-09 11:30:00 | 467.15 | 2024-12-11 09:15:00 | 474.30 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-12-09 14:00:00 | 467.35 | 2024-12-11 09:15:00 | 474.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-10 09:15:00 | 459.80 | 2024-12-11 09:15:00 | 474.30 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-12-12 12:00:00 | 467.90 | 2024-12-12 15:15:00 | 473.75 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-12-17 15:15:00 | 466.50 | 2024-12-19 09:15:00 | 443.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 15:15:00 | 466.50 | 2024-12-30 13:15:00 | 419.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-31 13:00:00 | 463.05 | 2025-01-31 15:15:00 | 478.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-01-31 13:45:00 | 465.50 | 2025-01-31 15:15:00 | 478.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-02-01 12:15:00 | 445.60 | 2025-02-03 09:15:00 | 423.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 15:15:00 | 431.90 | 2025-02-03 09:15:00 | 410.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 12:15:00 | 445.60 | 2025-02-03 11:15:00 | 401.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 15:15:00 | 431.90 | 2025-02-10 09:15:00 | 388.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-19 10:15:00 | 433.05 | 2025-05-20 14:15:00 | 411.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 10:15:00 | 433.05 | 2025-05-20 14:15:00 | 414.50 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2025-05-19 10:45:00 | 432.75 | 2025-05-21 09:15:00 | 411.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 10:45:00 | 432.75 | 2025-05-21 09:15:00 | 417.15 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-05-19 12:30:00 | 431.50 | 2025-05-21 09:15:00 | 409.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 12:30:00 | 431.50 | 2025-05-21 09:15:00 | 417.15 | STOP_HIT | 0.50 | 3.33% |
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
