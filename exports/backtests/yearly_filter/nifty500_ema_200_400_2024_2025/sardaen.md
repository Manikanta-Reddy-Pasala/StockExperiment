# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 591.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 2 |
| TARGET_HIT | 8 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 38
- **Target hits / Stop hits / Partials:** 8 / 39 / 2
- **Avg / median % per leg:** -0.70% / -2.19%
- **Sum % (uncompounded):** -34.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 8 | 22.9% | 8 | 27 | 0 | -0.20% | -6.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 8 | 22.9% | 8 | 27 | 0 | -0.20% | -6.9% |
| SELL (all) | 14 | 3 | 21.4% | 0 | 12 | 2 | -1.96% | -27.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 12 | 2 | -1.96% | -27.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 11 | 22.4% | 8 | 39 | 2 | -0.70% | -34.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 226.77 | 235.42 | 235.42 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 253.80 | 235.33 | 235.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 254.73 | 236.36 | 235.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 12:15:00 | 263.00 | 263.27 | 253.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 12:30:00 | 262.65 | 263.27 | 253.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 252.50 | 263.08 | 253.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 252.50 | 263.08 | 253.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 252.40 | 262.97 | 253.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:15:00 | 253.20 | 262.97 | 253.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 257.45 | 262.42 | 253.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 10:45:00 | 253.40 | 262.29 | 253.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 14:15:00 | 245.85 | 261.78 | 253.54 | SL hit (close<static) qty=1.00 sl=247.30 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 430.25 | 450.79 | 450.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 14:15:00 | 424.65 | 450.33 | 450.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 15:15:00 | 453.85 | 449.67 | 450.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 15:15:00 | 453.85 | 449.67 | 450.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 453.85 | 449.67 | 450.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:30:00 | 453.65 | 449.72 | 450.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 457.90 | 449.80 | 450.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:15:00 | 458.15 | 449.80 | 450.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 472.70 | 450.93 | 450.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 474.75 | 451.56 | 451.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 450.80 | 452.79 | 451.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 11:15:00 | 450.80 | 452.79 | 451.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 450.80 | 452.79 | 451.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 450.80 | 452.79 | 451.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 449.65 | 452.76 | 451.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:45:00 | 449.50 | 452.76 | 451.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 453.20 | 452.73 | 451.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 453.20 | 452.73 | 451.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 452.00 | 452.72 | 451.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 452.50 | 452.72 | 451.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 454.05 | 452.73 | 451.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 460.70 | 452.71 | 451.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:45:00 | 463.55 | 452.84 | 451.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:00:00 | 456.35 | 454.04 | 452.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 458.95 | 454.10 | 452.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 430.20 | 455.01 | 453.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 430.20 | 455.01 | 453.21 | SL hit (close<static) qty=1.00 sl=451.05 alert=retest2 |

### Cycle 5 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 437.60 | 480.57 | 480.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 433.00 | 477.89 | 479.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 467.00 | 462.86 | 470.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 10:00:00 | 467.00 | 462.86 | 470.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 462.40 | 462.85 | 470.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 460.95 | 463.61 | 470.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 475.60 | 463.81 | 470.14 | SL hit (close>static) qty=1.00 sl=472.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 541.20 | 451.02 | 450.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 549.90 | 455.48 | 453.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 568.80 | 571.98 | 539.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 568.80 | 571.98 | 539.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 548.50 | 573.69 | 547.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 548.50 | 573.69 | 547.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 547.30 | 573.43 | 547.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:45:00 | 547.30 | 573.43 | 547.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 544.90 | 573.14 | 547.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 544.90 | 573.14 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 540.60 | 572.82 | 547.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 540.60 | 572.82 | 547.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 542.40 | 571.60 | 547.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 542.40 | 571.60 | 547.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 550.80 | 570.62 | 547.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 554.15 | 570.42 | 547.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 546.40 | 569.78 | 552.33 | SL hit (close<static) qty=1.00 sl=546.65 alert=retest2 |

### Cycle 7 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 518.75 | 544.58 | 544.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 515.95 | 544.30 | 544.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 507.85 | 502.32 | 516.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 11:30:00 | 511.20 | 502.32 | 516.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 506.50 | 502.88 | 516.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 500.55 | 503.83 | 516.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 500.75 | 503.83 | 516.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:00:00 | 501.40 | 503.76 | 515.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 517.80 | 504.21 | 515.76 | SL hit (close>static) qty=1.00 sl=517.30 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 534.00 | 505.57 | 505.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 536.95 | 506.99 | 506.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:15:00 | 510.45 | 524.56 | 516.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 508.30 | 524.40 | 516.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 508.05 | 524.40 | 516.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 505.70 | 524.21 | 516.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 505.70 | 524.21 | 516.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 519.30 | 523.10 | 516.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:30:00 | 522.55 | 523.09 | 516.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 510.75 | 523.10 | 516.92 | SL hit (close<static) qty=1.00 sl=515.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-08-05 12:15:00 | 253.20 | 2024-08-06 14:15:00 | 245.85 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-08-06 09:15:00 | 257.45 | 2024-08-06 14:15:00 | 245.85 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest2 | 2024-08-06 10:45:00 | 253.40 | 2024-08-06 14:15:00 | 245.85 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-08-07 11:45:00 | 259.80 | 2024-08-08 14:15:00 | 285.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-14 09:30:00 | 433.30 | 2024-11-18 15:15:00 | 420.60 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-14 13:00:00 | 433.60 | 2024-11-18 15:15:00 | 420.60 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-11-25 09:15:00 | 435.80 | 2024-12-03 10:15:00 | 475.15 | TARGET_HIT | 1.00 | 9.03% |
| BUY | retest2 | 2024-11-25 10:15:00 | 433.95 | 2024-12-04 10:15:00 | 477.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 11:45:00 | 431.95 | 2024-12-04 11:15:00 | 479.38 | TARGET_HIT | 1.00 | 10.98% |
| BUY | retest2 | 2025-01-13 15:15:00 | 440.00 | 2025-01-14 11:15:00 | 423.95 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2025-01-14 10:00:00 | 430.70 | 2025-01-14 11:15:00 | 423.95 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-01-15 09:15:00 | 433.50 | 2025-01-27 09:15:00 | 409.70 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest2 | 2025-02-05 09:15:00 | 460.70 | 2025-02-11 09:15:00 | 430.20 | STOP_HIT | 1.00 | -6.62% |
| BUY | retest2 | 2025-02-05 10:45:00 | 463.55 | 2025-02-11 09:15:00 | 430.20 | STOP_HIT | 1.00 | -7.19% |
| BUY | retest2 | 2025-02-07 10:00:00 | 456.35 | 2025-02-11 09:15:00 | 430.20 | STOP_HIT | 1.00 | -5.73% |
| BUY | retest2 | 2025-02-07 10:45:00 | 458.95 | 2025-02-11 09:15:00 | 430.20 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest2 | 2025-02-11 11:15:00 | 440.30 | 2025-02-17 14:15:00 | 479.71 | TARGET_HIT | 1.00 | 8.95% |
| BUY | retest2 | 2025-02-12 09:45:00 | 436.10 | 2025-02-17 15:15:00 | 484.33 | TARGET_HIT | 1.00 | 11.06% |
| BUY | retest2 | 2025-04-11 09:15:00 | 435.45 | 2025-04-11 12:15:00 | 479.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-05 11:15:00 | 441.45 | 2025-05-06 09:15:00 | 437.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-05-21 09:15:00 | 460.95 | 2025-05-21 10:15:00 | 475.60 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-05-26 09:15:00 | 437.65 | 2025-07-14 15:15:00 | 437.38 | PARTIAL | 0.50 | 0.06% |
| SELL | retest2 | 2025-05-26 09:15:00 | 437.65 | 2025-07-15 09:15:00 | 442.20 | STOP_HIT | 0.50 | -1.04% |
| SELL | retest2 | 2025-07-11 10:00:00 | 460.40 | 2025-08-04 09:15:00 | 512.85 | STOP_HIT | 1.00 | -11.39% |
| BUY | retest2 | 2025-10-03 09:15:00 | 554.15 | 2025-10-14 09:15:00 | 546.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-16 09:30:00 | 553.00 | 2025-10-16 10:15:00 | 546.15 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-23 09:15:00 | 556.50 | 2025-10-23 09:15:00 | 543.05 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-10-29 09:15:00 | 558.00 | 2025-10-30 10:15:00 | 544.55 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-11-10 14:15:00 | 546.70 | 2025-11-11 09:15:00 | 534.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-11-10 15:00:00 | 543.20 | 2025-11-11 09:15:00 | 534.70 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-11-11 09:45:00 | 542.35 | 2025-11-11 10:15:00 | 531.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-12-17 10:15:00 | 500.55 | 2025-12-19 09:15:00 | 517.80 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-12-18 09:15:00 | 500.75 | 2025-12-19 09:15:00 | 517.80 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-12-18 11:00:00 | 501.40 | 2025-12-19 09:15:00 | 517.80 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2026-01-08 10:00:00 | 501.20 | 2026-01-09 15:15:00 | 476.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 501.20 | 2026-01-30 15:15:00 | 492.00 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2026-02-09 09:45:00 | 507.05 | 2026-02-09 10:15:00 | 513.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-02-11 09:15:00 | 503.90 | 2026-02-16 12:15:00 | 504.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-02-12 12:30:00 | 506.40 | 2026-02-17 13:15:00 | 515.05 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-12 14:15:00 | 507.80 | 2026-02-17 13:15:00 | 515.05 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-16 09:15:00 | 494.40 | 2026-02-17 13:15:00 | 515.05 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-17 14:30:00 | 522.55 | 2026-03-19 13:15:00 | 510.75 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-20 09:15:00 | 522.90 | 2026-03-20 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-20 10:30:00 | 521.55 | 2026-03-20 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-03-25 09:15:00 | 522.60 | 2026-03-27 09:15:00 | 510.45 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-04-01 09:15:00 | 529.60 | 2026-04-02 09:15:00 | 506.15 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2026-04-01 12:00:00 | 526.95 | 2026-04-02 09:15:00 | 506.15 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2026-04-01 15:00:00 | 526.30 | 2026-04-02 09:15:00 | 506.15 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-04-07 09:15:00 | 527.80 | 2026-04-13 10:15:00 | 580.58 | TARGET_HIT | 1.00 | 10.00% |
