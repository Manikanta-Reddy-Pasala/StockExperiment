# Leela Palaces Hotels & Resorts Ltd. (THELEELA)

## Backtest Summary

- **Window:** 2025-06-02 09:15:00 → 2026-05-11 15:15:00 (1626 bars)
- **Last close:** 413.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 46 |
| ALERT2 | 44 |
| ALERT2_SKIP | 24 |
| ALERT3 | 125 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 45 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 38
- **Target hits / Stop hits / Partials:** 1 / 46 / 3
- **Avg / median % per leg:** -0.88% / -1.54%
- **Sum % (uncompounded):** -43.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 4 | 15.4% | 1 | 25 | 0 | -1.09% | -28.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.41% | -2.8% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 1 | 23 | 0 | -1.07% | -25.6% |
| SELL (all) | 24 | 8 | 33.3% | 0 | 21 | 3 | -0.65% | -15.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 8 | 33.3% | 0 | 21 | 3 | -0.65% | -15.5% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.41% | -2.8% |
| retest2 (combined) | 48 | 12 | 25.0% | 1 | 44 | 3 | -0.86% | -41.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 432.00 | 433.20 | 433.22 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 434.75 | 432.39 | 432.30 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 10:15:00 | 427.35 | 431.64 | 432.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 11:15:00 | 422.80 | 429.87 | 431.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 400.85 | 399.83 | 406.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 400.30 | 399.83 | 406.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 404.45 | 400.96 | 404.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 405.95 | 400.96 | 404.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 406.30 | 402.02 | 404.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 406.05 | 402.02 | 404.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 410.45 | 403.71 | 405.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 410.45 | 403.71 | 405.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 410.55 | 406.16 | 406.05 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 397.90 | 405.11 | 405.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 395.60 | 403.21 | 404.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 392.10 | 390.98 | 394.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 14:00:00 | 392.10 | 390.98 | 394.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 392.90 | 391.37 | 393.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 392.90 | 391.37 | 393.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 393.75 | 391.84 | 393.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 387.50 | 391.84 | 393.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 400.85 | 391.54 | 392.13 | SL hit (close>static) qty=1.00 sl=395.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 400.45 | 393.32 | 392.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 402.75 | 399.77 | 396.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 400.30 | 402.51 | 399.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 400.30 | 402.51 | 399.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 400.30 | 402.51 | 399.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 400.30 | 402.51 | 399.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 401.75 | 401.96 | 400.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 401.80 | 401.96 | 400.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 403.10 | 402.15 | 400.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 402.30 | 402.15 | 400.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 406.00 | 403.07 | 401.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:00:00 | 407.00 | 404.68 | 402.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:00:00 | 406.20 | 404.98 | 402.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 409.25 | 406.01 | 403.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 397.95 | 404.60 | 403.58 | SL hit (close<static) qty=1.00 sl=398.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 402.65 | 405.30 | 405.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 401.95 | 403.55 | 404.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 402.50 | 402.49 | 403.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 402.50 | 402.49 | 403.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 402.50 | 402.49 | 403.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 401.45 | 402.49 | 403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 405.80 | 403.15 | 403.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 405.50 | 403.15 | 403.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 406.00 | 403.72 | 404.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 405.00 | 403.72 | 404.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 406.00 | 404.53 | 404.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 425.80 | 408.78 | 406.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 10:15:00 | 422.30 | 422.61 | 416.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:00:00 | 422.30 | 422.61 | 416.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 435.00 | 435.15 | 432.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:45:00 | 440.00 | 436.98 | 435.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:45:00 | 439.70 | 437.77 | 436.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 450.00 | 452.54 | 452.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 450.00 | 452.54 | 452.87 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 457.85 | 453.61 | 453.32 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 447.95 | 456.29 | 457.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 445.60 | 450.81 | 453.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 423.55 | 422.36 | 430.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:30:00 | 422.25 | 422.36 | 430.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 422.85 | 418.00 | 419.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 422.85 | 418.00 | 419.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 426.85 | 419.77 | 420.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 426.85 | 419.77 | 420.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 425.40 | 420.89 | 420.58 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 415.50 | 421.33 | 422.03 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 09:15:00 | 426.25 | 422.19 | 422.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 429.85 | 425.13 | 423.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 15:15:00 | 430.65 | 433.57 | 429.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:15:00 | 432.00 | 433.57 | 429.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 429.25 | 432.71 | 429.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 430.55 | 432.71 | 429.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 430.20 | 432.20 | 429.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 429.40 | 432.20 | 429.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 428.00 | 431.36 | 429.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 427.35 | 431.36 | 429.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 428.50 | 430.79 | 429.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 426.80 | 430.79 | 429.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 427.50 | 430.05 | 429.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 427.50 | 430.05 | 429.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 427.50 | 429.54 | 429.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 436.50 | 429.54 | 429.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 434.35 | 440.09 | 440.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 434.35 | 440.09 | 440.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 433.00 | 438.67 | 440.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 402.95 | 401.67 | 407.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 402.95 | 401.67 | 407.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 402.65 | 399.66 | 403.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 405.00 | 399.66 | 403.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 405.80 | 400.88 | 403.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 405.30 | 400.88 | 403.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 406.40 | 401.99 | 404.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 406.40 | 401.99 | 404.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 407.20 | 404.07 | 404.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 412.30 | 404.07 | 404.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 417.65 | 406.78 | 405.77 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 407.40 | 412.62 | 412.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 405.20 | 411.13 | 412.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 11:15:00 | 411.45 | 411.20 | 412.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 11:15:00 | 411.45 | 411.20 | 412.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 411.45 | 411.20 | 412.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 411.65 | 411.20 | 412.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 412.30 | 411.42 | 412.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 412.15 | 411.42 | 412.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 412.75 | 411.68 | 412.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 412.75 | 411.68 | 412.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 412.05 | 411.76 | 412.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 413.90 | 411.76 | 412.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 414.00 | 412.21 | 412.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 405.60 | 412.21 | 412.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 411.25 | 412.01 | 412.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:30:00 | 404.50 | 408.75 | 410.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 404.05 | 406.55 | 408.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 404.00 | 403.57 | 405.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 405.05 | 403.86 | 405.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 414.90 | 406.07 | 406.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 414.90 | 406.07 | 406.40 | SL hit (close>static) qty=1.00 sl=414.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 416.15 | 408.08 | 407.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 429.05 | 416.11 | 412.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 418.45 | 419.02 | 416.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 414.00 | 419.02 | 416.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 419.80 | 419.18 | 416.44 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 411.15 | 415.73 | 416.14 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 422.20 | 417.29 | 416.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 423.45 | 418.52 | 417.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 13:15:00 | 423.85 | 424.65 | 421.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:00:00 | 423.85 | 424.65 | 421.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 422.00 | 424.12 | 421.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 422.70 | 424.12 | 421.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 426.90 | 424.68 | 422.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 425.65 | 424.68 | 422.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 430.50 | 425.84 | 423.14 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 417.40 | 422.11 | 422.41 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 425.70 | 422.82 | 422.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 13:15:00 | 428.95 | 425.03 | 423.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 432.85 | 434.33 | 430.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:45:00 | 433.20 | 434.33 | 430.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 433.00 | 434.06 | 431.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:45:00 | 431.50 | 434.06 | 431.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 429.00 | 433.05 | 430.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 429.00 | 433.05 | 430.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 430.85 | 432.61 | 430.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 430.20 | 432.61 | 430.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 430.20 | 432.13 | 430.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 430.70 | 432.13 | 430.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 430.20 | 431.74 | 430.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 430.20 | 431.74 | 430.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 430.00 | 431.39 | 430.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 432.80 | 431.39 | 430.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 437.15 | 432.55 | 431.30 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 423.90 | 431.03 | 431.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 420.45 | 427.73 | 429.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 11:15:00 | 423.95 | 421.17 | 424.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 423.95 | 421.17 | 424.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 422.10 | 421.35 | 424.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 416.20 | 420.07 | 423.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 416.10 | 416.54 | 420.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 428.85 | 420.24 | 419.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 428.85 | 420.24 | 419.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 433.45 | 422.88 | 421.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 11:15:00 | 426.25 | 428.01 | 425.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 11:15:00 | 426.25 | 428.01 | 425.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 426.25 | 428.01 | 425.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 427.00 | 428.01 | 425.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 430.70 | 428.55 | 425.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:00:00 | 432.25 | 429.29 | 426.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 438.60 | 429.84 | 427.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 425.20 | 428.39 | 427.63 | SL hit (close<static) qty=1.00 sl=425.50 alert=retest2 |

### Cycle 25 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 419.25 | 426.16 | 426.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 418.15 | 421.77 | 423.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 425.55 | 422.52 | 424.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 425.55 | 422.52 | 424.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 425.55 | 422.52 | 424.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 428.20 | 422.52 | 424.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 427.10 | 423.44 | 424.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 428.70 | 423.44 | 424.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 425.65 | 424.10 | 424.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 425.70 | 424.10 | 424.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 428.95 | 425.07 | 424.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 14:15:00 | 430.40 | 426.14 | 425.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 14:15:00 | 443.55 | 459.73 | 455.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 443.55 | 459.73 | 455.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 443.55 | 459.73 | 455.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 431.55 | 459.73 | 455.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 434.00 | 454.59 | 453.97 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 435.90 | 450.85 | 452.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 10:15:00 | 432.95 | 447.27 | 450.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 448.10 | 443.09 | 446.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 448.10 | 443.09 | 446.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 448.10 | 443.09 | 446.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 456.35 | 443.09 | 446.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 449.35 | 444.34 | 446.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 449.35 | 444.34 | 446.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 447.55 | 444.98 | 446.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:00:00 | 443.70 | 444.73 | 446.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 445.90 | 443.52 | 444.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:15:00 | 423.60 | 428.73 | 430.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 430.25 | 429.04 | 430.51 | SL hit (close>ema200) qty=0.50 sl=429.04 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 434.85 | 430.82 | 430.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 439.55 | 432.57 | 431.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 434.50 | 437.56 | 435.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 434.50 | 437.56 | 435.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 434.50 | 437.56 | 435.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 434.50 | 437.56 | 435.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 431.95 | 436.44 | 435.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 440.65 | 436.44 | 435.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 435.70 | 436.45 | 435.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 435.70 | 436.45 | 435.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 435.15 | 436.19 | 435.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 435.15 | 436.19 | 435.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 436.70 | 436.29 | 435.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:00:00 | 439.70 | 436.98 | 435.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 435.00 | 436.58 | 435.85 | SL hit (close<static) qty=1.00 sl=435.15 alert=retest2 |

### Cycle 29 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 422.95 | 435.55 | 436.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 418.00 | 428.62 | 432.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 425.80 | 422.54 | 426.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 425.80 | 422.54 | 426.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 430.65 | 424.16 | 427.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 429.60 | 424.16 | 427.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 439.55 | 427.24 | 428.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 439.55 | 427.24 | 428.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 438.95 | 429.58 | 429.20 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 428.45 | 428.86 | 428.91 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 431.80 | 429.45 | 429.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 435.65 | 430.69 | 429.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 433.70 | 434.44 | 432.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 433.70 | 434.44 | 432.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 433.70 | 434.44 | 432.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 431.40 | 434.44 | 432.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 432.10 | 433.97 | 432.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 432.10 | 433.97 | 432.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 428.85 | 432.95 | 431.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 428.85 | 432.95 | 431.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 434.75 | 433.31 | 432.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:15:00 | 435.45 | 433.64 | 432.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 437.85 | 432.69 | 432.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:15:00 | 437.00 | 433.27 | 432.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 435.35 | 434.66 | 433.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 433.60 | 434.72 | 433.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 431.25 | 434.72 | 433.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 431.10 | 433.99 | 433.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 431.10 | 433.99 | 433.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 431.80 | 433.55 | 433.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 431.50 | 433.14 | 433.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 431.50 | 433.14 | 433.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 429.25 | 432.36 | 432.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 435.05 | 431.54 | 432.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 435.05 | 431.54 | 432.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 435.05 | 431.54 | 432.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 435.50 | 431.54 | 432.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 435.15 | 432.26 | 432.45 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 437.05 | 433.22 | 432.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 438.00 | 434.18 | 433.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 10:15:00 | 432.55 | 435.12 | 434.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 10:15:00 | 432.55 | 435.12 | 434.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 432.55 | 435.12 | 434.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 432.55 | 435.12 | 434.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 430.10 | 434.12 | 433.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 430.10 | 434.12 | 433.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 430.95 | 433.48 | 433.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 428.95 | 431.74 | 432.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 428.20 | 427.28 | 429.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 428.20 | 427.28 | 429.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 428.20 | 427.28 | 429.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 428.20 | 427.28 | 429.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 429.85 | 427.79 | 429.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:15:00 | 430.00 | 427.79 | 429.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 430.05 | 428.24 | 429.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 430.00 | 428.24 | 429.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 409.90 | 410.25 | 413.25 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 416.85 | 413.28 | 413.28 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 412.35 | 413.10 | 413.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 408.00 | 412.08 | 412.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 411.65 | 410.85 | 411.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 411.65 | 410.85 | 411.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 411.65 | 410.85 | 411.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 411.60 | 410.85 | 411.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 411.60 | 411.00 | 411.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 411.35 | 411.00 | 411.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 416.25 | 412.05 | 412.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 416.25 | 412.05 | 412.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 415.05 | 412.65 | 412.40 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 407.20 | 412.57 | 413.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 404.60 | 410.33 | 412.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 407.55 | 407.50 | 409.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:30:00 | 408.95 | 407.50 | 409.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 409.00 | 406.69 | 408.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 400.90 | 405.41 | 407.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 398.90 | 393.39 | 394.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 396.45 | 393.62 | 393.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 396.45 | 393.62 | 393.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 400.00 | 394.89 | 394.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 396.80 | 397.18 | 395.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 15:15:00 | 396.80 | 397.18 | 395.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 396.80 | 397.18 | 395.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 400.25 | 395.94 | 395.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-26 14:15:00 | 440.28 | 431.70 | 425.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 418.10 | 424.74 | 424.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 416.40 | 422.07 | 423.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 434.35 | 423.60 | 423.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 434.35 | 423.60 | 423.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 434.35 | 423.60 | 423.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 434.35 | 423.60 | 423.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 430.90 | 425.06 | 424.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 441.45 | 435.55 | 432.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 442.15 | 442.21 | 438.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 442.15 | 442.21 | 438.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 435.30 | 440.83 | 438.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 436.00 | 440.83 | 438.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 433.00 | 439.26 | 438.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 433.00 | 439.26 | 438.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 430.30 | 436.01 | 436.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 12:15:00 | 424.00 | 430.96 | 434.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 11:15:00 | 423.65 | 423.56 | 428.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 423.65 | 423.56 | 428.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 427.60 | 422.75 | 425.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 427.70 | 422.75 | 425.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 422.55 | 422.71 | 425.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 427.85 | 422.71 | 425.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 425.00 | 421.43 | 423.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 425.00 | 421.43 | 423.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 423.05 | 421.75 | 423.11 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 14:15:00 | 430.80 | 424.16 | 424.01 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 421.80 | 423.94 | 423.97 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 12:15:00 | 425.35 | 424.23 | 424.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 13:15:00 | 430.60 | 425.51 | 424.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 445.15 | 446.12 | 440.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 445.15 | 446.12 | 440.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 445.15 | 446.12 | 440.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 13:30:00 | 469.00 | 446.26 | 442.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 454.80 | 446.26 | 442.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 439.05 | 444.82 | 441.94 | SL hit (close<static) qty=1.00 sl=440.00 alert=retest2 |

### Cycle 47 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 433.35 | 439.27 | 439.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 413.25 | 433.16 | 436.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 407.50 | 407.41 | 413.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:00:00 | 407.50 | 407.41 | 413.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 404.55 | 406.12 | 411.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 399.70 | 405.97 | 409.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 401.70 | 406.15 | 407.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 417.85 | 407.11 | 406.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 417.85 | 407.11 | 406.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 420.20 | 411.15 | 408.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 422.10 | 423.65 | 418.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 415.45 | 423.65 | 418.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 423.50 | 423.62 | 418.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 418.20 | 423.62 | 418.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 420.90 | 422.61 | 419.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 425.00 | 423.09 | 419.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 430.65 | 436.42 | 437.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 430.65 | 436.42 | 437.14 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 442.60 | 436.89 | 436.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 448.70 | 441.41 | 438.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 440.35 | 442.46 | 440.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 449.35 | 444.01 | 441.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 14:15:00 | 449.00 | 445.49 | 442.76 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 446.35 | 447.83 | 444.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 437.15 | 447.83 | 444.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 446.50 | 447.56 | 444.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 444.60 | 447.56 | 444.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 442.85 | 446.62 | 444.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 442.85 | 446.62 | 444.68 | SL hit (close<ema400) qty=1.00 sl=444.68 alert=retest1 |

### Cycle 51 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 441.30 | 443.60 | 443.86 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 447.40 | 444.22 | 444.08 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 439.10 | 443.34 | 443.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 437.25 | 442.12 | 443.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 441.95 | 439.99 | 441.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 15:15:00 | 441.95 | 439.99 | 441.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 441.95 | 439.99 | 441.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 436.20 | 439.99 | 441.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 442.45 | 440.29 | 441.08 | SL hit (close>static) qty=1.00 sl=442.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 443.45 | 441.73 | 441.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 444.85 | 442.35 | 441.80 | Break + close above crossover candle high |

### Cycle 55 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 435.00 | 440.99 | 441.33 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 439.00 | 436.86 | 436.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 15:15:00 | 440.50 | 439.28 | 438.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 441.15 | 441.26 | 439.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 441.15 | 441.26 | 439.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 441.15 | 441.26 | 439.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:15:00 | 438.60 | 441.26 | 439.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 438.60 | 440.73 | 439.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 431.85 | 440.73 | 439.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 435.70 | 439.72 | 439.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 457.15 | 443.55 | 441.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 437.35 | 441.62 | 441.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 437.35 | 441.62 | 441.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 435.75 | 438.45 | 439.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 434.50 | 433.80 | 436.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 15:00:00 | 434.50 | 433.80 | 436.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 440.00 | 430.16 | 432.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 425.40 | 429.51 | 431.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 438.80 | 432.24 | 431.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 438.80 | 432.24 | 431.97 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 427.75 | 431.35 | 431.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 425.65 | 430.21 | 431.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 14:15:00 | 435.05 | 431.14 | 431.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 435.05 | 431.14 | 431.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 435.05 | 431.14 | 431.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 435.05 | 431.14 | 431.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 429.90 | 430.89 | 431.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 417.20 | 430.89 | 431.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 12:15:00 | 435.00 | 431.16 | 431.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 435.00 | 431.16 | 431.06 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 427.00 | 430.53 | 430.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 424.45 | 429.31 | 430.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 429.35 | 429.32 | 430.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:00:00 | 429.35 | 429.32 | 430.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 432.70 | 430.00 | 430.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 432.70 | 430.00 | 430.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 432.00 | 430.40 | 430.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 420.30 | 430.40 | 430.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 399.28 | 403.50 | 407.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 393.90 | 393.33 | 399.36 | SL hit (close>ema200) qty=0.50 sl=393.33 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 404.90 | 399.74 | 399.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 409.90 | 401.77 | 400.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 410.15 | 411.76 | 408.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 13:15:00 | 410.15 | 411.76 | 408.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 410.15 | 411.76 | 408.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:45:00 | 409.00 | 411.76 | 408.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 409.50 | 411.31 | 408.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:45:00 | 409.00 | 411.31 | 408.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 401.30 | 409.31 | 408.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 402.15 | 409.31 | 408.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 401.45 | 407.74 | 407.52 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 398.35 | 405.86 | 406.68 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 417.00 | 407.70 | 406.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 420.00 | 415.68 | 414.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 414.00 | 417.01 | 415.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 14:15:00 | 414.00 | 417.01 | 415.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 414.00 | 417.01 | 415.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 414.00 | 417.01 | 415.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 414.00 | 416.41 | 415.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 437.00 | 416.41 | 415.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 427.95 | 429.38 | 429.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 427.95 | 429.38 | 429.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 426.25 | 428.75 | 429.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 428.80 | 428.38 | 428.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 428.80 | 428.38 | 428.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 428.80 | 428.38 | 428.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:30:00 | 428.00 | 428.38 | 428.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 425.15 | 427.73 | 428.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:00:00 | 424.00 | 426.99 | 428.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 422.35 | 426.51 | 427.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 10:45:00 | 423.80 | 426.86 | 427.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 430.55 | 427.60 | 427.80 | SL hit (close>static) qty=1.00 sl=429.30 alert=retest2 |

### Cycle 66 — BUY (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 12:15:00 | 430.75 | 428.23 | 428.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 15:15:00 | 432.00 | 429.50 | 428.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 10:15:00 | 433.15 | 433.78 | 431.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:30:00 | 433.65 | 433.78 | 431.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 433.15 | 433.62 | 432.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:30:00 | 433.00 | 433.62 | 432.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 432.55 | 433.41 | 432.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 432.55 | 433.41 | 432.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 432.50 | 433.08 | 432.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 433.00 | 433.08 | 432.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 432.50 | 432.97 | 432.27 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 424.40 | 430.96 | 431.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 10:15:00 | 418.05 | 425.90 | 428.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 14:15:00 | 424.65 | 420.48 | 424.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 14:15:00 | 424.65 | 420.48 | 424.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 424.65 | 420.48 | 424.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 424.65 | 420.48 | 424.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 419.90 | 420.37 | 424.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 427.60 | 420.37 | 424.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 430.45 | 422.38 | 424.68 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 429.45 | 426.38 | 426.16 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 418.50 | 424.87 | 425.53 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 428.35 | 424.67 | 424.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 429.60 | 425.66 | 424.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 427.20 | 427.31 | 425.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 427.20 | 427.31 | 425.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 427.20 | 427.31 | 425.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:15:00 | 424.70 | 427.31 | 425.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 424.45 | 426.74 | 425.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 424.25 | 426.74 | 425.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 424.30 | 426.25 | 425.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 424.10 | 426.25 | 425.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 419.05 | 424.81 | 425.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 09:15:00 | 412.25 | 419.72 | 420.95 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-23 09:15:00 | 387.50 | 2025-06-24 09:15:00 | 400.85 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-06-27 13:00:00 | 407.00 | 2025-06-30 10:15:00 | 397.95 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-06-27 14:00:00 | 406.20 | 2025-06-30 10:15:00 | 397.95 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-27 14:30:00 | 409.25 | 2025-06-30 10:15:00 | 397.95 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-06-30 15:15:00 | 407.40 | 2025-07-01 15:15:00 | 401.30 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-01 13:15:00 | 404.00 | 2025-07-04 11:15:00 | 402.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-07-02 09:15:00 | 404.00 | 2025-07-04 11:15:00 | 402.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-07-16 09:45:00 | 440.00 | 2025-07-22 15:15:00 | 450.00 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2025-07-16 12:45:00 | 439.70 | 2025-07-22 15:15:00 | 450.00 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-08-12 09:15:00 | 436.50 | 2025-08-20 14:15:00 | 434.35 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-09-08 13:30:00 | 404.50 | 2025-09-10 12:15:00 | 414.90 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-09-09 09:30:00 | 404.05 | 2025-09-10 12:15:00 | 414.90 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-09-10 10:30:00 | 404.00 | 2025-09-10 12:15:00 | 414.90 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-09-10 11:30:00 | 405.05 | 2025-09-10 12:15:00 | 414.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-09-29 15:00:00 | 416.20 | 2025-10-01 11:15:00 | 428.85 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-09-30 10:30:00 | 416.10 | 2025-10-01 11:15:00 | 428.85 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-10-03 14:00:00 | 432.25 | 2025-10-06 14:15:00 | 425.20 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-10-06 09:15:00 | 438.60 | 2025-10-06 14:15:00 | 425.20 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-10-16 13:00:00 | 443.70 | 2025-10-29 09:15:00 | 423.60 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2025-10-16 13:00:00 | 443.70 | 2025-10-29 10:15:00 | 430.25 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-10-17 10:45:00 | 445.90 | 2025-10-30 09:15:00 | 421.51 | PARTIAL | 0.50 | 5.47% |
| SELL | retest2 | 2025-10-17 10:45:00 | 445.90 | 2025-10-30 09:15:00 | 435.00 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest2 | 2025-11-03 14:00:00 | 439.70 | 2025-11-03 14:15:00 | 435.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-11-04 09:15:00 | 439.10 | 2025-11-06 09:15:00 | 422.95 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-11-11 14:15:00 | 435.45 | 2025-11-14 11:15:00 | 431.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-13 09:15:00 | 437.85 | 2025-11-14 11:15:00 | 431.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-13 11:15:00 | 437.00 | 2025-11-14 11:15:00 | 431.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-13 12:30:00 | 435.35 | 2025-11-14 11:15:00 | 431.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-05 09:30:00 | 400.90 | 2025-12-12 10:15:00 | 396.45 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-12-10 10:00:00 | 398.90 | 2025-12-12 10:15:00 | 396.45 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-12-16 09:15:00 | 400.25 | 2025-12-26 14:15:00 | 440.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-16 13:30:00 | 469.00 | 2026-01-16 14:15:00 | 439.05 | STOP_HIT | 1.00 | -6.39% |
| BUY | retest2 | 2026-01-16 14:15:00 | 454.80 | 2026-01-16 14:15:00 | 439.05 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2026-01-27 09:15:00 | 399.70 | 2026-01-30 11:15:00 | 417.85 | STOP_HIT | 1.00 | -4.54% |
| SELL | retest2 | 2026-01-29 10:15:00 | 401.70 | 2026-01-30 11:15:00 | 417.85 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2026-02-02 13:00:00 | 425.00 | 2026-02-06 09:15:00 | 430.65 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest1 | 2026-02-11 10:30:00 | 449.35 | 2026-02-12 11:15:00 | 442.85 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest1 | 2026-02-11 14:15:00 | 449.00 | 2026-02-12 11:15:00 | 442.85 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-12 14:30:00 | 445.55 | 2026-02-13 09:15:00 | 437.95 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-17 09:15:00 | 436.20 | 2026-02-17 13:15:00 | 442.45 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-18 09:15:00 | 437.05 | 2026-02-18 10:15:00 | 443.90 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-02-27 15:00:00 | 457.15 | 2026-03-04 11:15:00 | 437.35 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2026-03-10 15:00:00 | 425.40 | 2026-03-11 10:15:00 | 438.80 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-03-12 09:15:00 | 417.20 | 2026-03-12 12:15:00 | 435.00 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2026-03-16 09:15:00 | 420.30 | 2026-03-20 14:15:00 | 399.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:15:00 | 420.30 | 2026-03-23 14:15:00 | 393.90 | STOP_HIT | 0.50 | 6.28% |
| BUY | retest2 | 2026-04-08 09:15:00 | 437.00 | 2026-04-16 11:15:00 | 427.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-04-17 12:00:00 | 424.00 | 2026-04-20 11:15:00 | 430.55 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-04-20 09:30:00 | 422.35 | 2026-04-20 11:15:00 | 430.55 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-20 10:45:00 | 423.80 | 2026-04-20 11:15:00 | 430.55 | STOP_HIT | 1.00 | -1.59% |
