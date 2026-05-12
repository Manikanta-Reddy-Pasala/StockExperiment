# Rail Vikas Nigam Ltd. (RVNL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 305.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 55 |
| ALERT1 | 43 |
| ALERT2 | 41 |
| ALERT2_SKIP | 26 |
| ALERT3 | 100 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 37 / 30
- **Target hits / Stop hits / Partials:** 4 / 49 / 14
- **Avg / median % per leg:** 1.78% / 0.70%
- **Sum % (uncompounded):** 119.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.03% | 18.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.03% | 18.6% |
| SELL (all) | 49 | 33 | 67.3% | 0 | 35 | 14 | 2.06% | 101.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 49 | 33 | 67.3% | 0 | 35 | 14 | 2.06% | 101.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 67 | 37 | 55.2% | 4 | 49 | 14 | 1.78% | 119.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 349.20 | 336.54 | 335.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 360.40 | 345.99 | 340.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 418.20 | 424.36 | 410.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:30:00 | 419.35 | 424.36 | 410.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 414.50 | 419.23 | 412.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 417.15 | 418.17 | 413.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 409.85 | 415.96 | 413.25 | SL hit (close<static) qty=1.00 sl=411.30 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 406.65 | 411.88 | 412.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 404.60 | 408.79 | 410.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 409.75 | 402.36 | 404.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 409.75 | 402.36 | 404.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 409.75 | 402.36 | 404.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 415.70 | 402.36 | 404.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 408.85 | 403.66 | 405.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:30:00 | 407.45 | 404.96 | 405.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 409.45 | 406.50 | 406.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 409.45 | 406.50 | 406.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 412.25 | 407.65 | 406.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 414.60 | 416.83 | 414.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 15:15:00 | 414.60 | 416.83 | 414.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 414.60 | 416.83 | 414.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 412.15 | 416.83 | 414.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 412.65 | 415.99 | 414.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 412.20 | 415.99 | 414.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 413.00 | 415.39 | 413.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 412.55 | 415.39 | 413.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 415.50 | 415.60 | 414.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 415.05 | 415.60 | 414.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 414.15 | 415.31 | 414.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 414.15 | 415.31 | 414.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 415.90 | 415.43 | 414.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 419.30 | 415.43 | 414.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 413.60 | 415.47 | 414.79 | SL hit (close<static) qty=1.00 sl=414.10 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 411.80 | 414.21 | 414.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 409.90 | 413.35 | 413.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 406.85 | 406.55 | 409.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 406.85 | 406.55 | 409.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 406.85 | 406.55 | 409.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:00:00 | 403.35 | 405.95 | 407.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:30:00 | 403.90 | 405.54 | 407.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 403.90 | 405.54 | 407.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 403.90 | 405.27 | 407.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 405.50 | 405.10 | 406.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:15:00 | 413.45 | 405.10 | 406.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 417.30 | 407.54 | 407.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 417.30 | 407.54 | 407.76 | SL hit (close>static) qty=1.00 sl=411.40 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 429.65 | 411.96 | 409.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 432.30 | 416.03 | 411.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 421.60 | 427.04 | 422.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 421.60 | 427.04 | 422.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 421.60 | 427.04 | 422.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 421.85 | 427.04 | 422.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 421.75 | 425.98 | 422.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 421.75 | 425.98 | 422.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 424.25 | 425.63 | 422.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 429.95 | 426.36 | 423.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:45:00 | 426.85 | 429.51 | 428.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:00:00 | 427.90 | 428.20 | 427.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 428.35 | 427.78 | 427.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 432.15 | 428.66 | 428.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 423.00 | 427.41 | 427.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 423.00 | 427.41 | 427.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 418.85 | 425.70 | 427.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 409.90 | 407.49 | 411.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 409.90 | 407.49 | 411.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 410.25 | 408.79 | 411.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 410.55 | 408.79 | 411.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 407.15 | 408.47 | 411.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 405.35 | 407.81 | 410.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 385.08 | 392.71 | 398.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 388.00 | 387.33 | 393.32 | SL hit (close>ema200) qty=0.50 sl=387.33 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 398.90 | 392.75 | 392.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 404.00 | 400.51 | 399.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 11:15:00 | 400.65 | 400.96 | 399.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 12:00:00 | 400.65 | 400.96 | 399.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 398.65 | 400.50 | 399.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 398.65 | 400.50 | 399.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 398.15 | 400.03 | 399.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:15:00 | 397.85 | 400.03 | 399.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 395.10 | 399.04 | 399.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 395.10 | 399.04 | 399.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 396.30 | 398.49 | 398.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 392.70 | 395.32 | 396.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 393.95 | 393.29 | 394.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 10:15:00 | 393.95 | 393.29 | 394.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 393.95 | 393.29 | 394.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 392.35 | 393.08 | 394.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:15:00 | 392.15 | 393.08 | 394.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 391.75 | 392.82 | 393.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 392.25 | 392.19 | 393.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 390.05 | 391.73 | 392.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 388.60 | 389.79 | 390.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 388.20 | 387.48 | 388.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 387.45 | 386.38 | 387.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 372.73 | 375.04 | 376.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 372.54 | 375.04 | 376.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 372.16 | 375.04 | 376.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 372.64 | 375.04 | 376.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:15:00 | 369.17 | 373.96 | 376.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 10:15:00 | 368.79 | 372.86 | 375.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:15:00 | 368.08 | 372.01 | 374.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 12:15:00 | 372.90 | 372.19 | 374.58 | SL hit (close>ema200) qty=0.50 sl=372.19 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 378.25 | 375.32 | 375.08 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 371.15 | 375.00 | 375.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 369.90 | 373.98 | 374.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 357.90 | 357.88 | 362.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 357.90 | 357.88 | 362.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 361.10 | 358.71 | 361.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 360.15 | 358.71 | 361.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 358.00 | 358.57 | 361.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 357.15 | 358.57 | 361.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 354.00 | 358.85 | 360.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 354.65 | 352.27 | 351.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 354.65 | 352.27 | 351.97 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 346.95 | 351.15 | 351.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 345.55 | 350.03 | 351.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 346.80 | 343.74 | 346.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 346.80 | 343.74 | 346.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 346.80 | 343.74 | 346.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 346.80 | 343.74 | 346.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 347.20 | 344.43 | 346.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 345.15 | 344.43 | 346.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 344.70 | 343.31 | 343.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 344.70 | 343.31 | 343.28 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 327.25 | 340.37 | 341.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 322.00 | 333.92 | 338.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 328.90 | 327.86 | 332.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 328.90 | 327.86 | 332.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 328.90 | 327.86 | 332.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 326.95 | 327.50 | 331.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 326.00 | 326.21 | 328.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 324.95 | 326.25 | 326.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 331.15 | 327.65 | 327.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 331.15 | 327.65 | 327.31 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 324.10 | 328.16 | 328.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 322.85 | 324.20 | 325.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 317.25 | 315.98 | 319.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 317.25 | 315.98 | 319.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 308.10 | 306.46 | 310.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 308.90 | 306.46 | 310.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 311.15 | 307.77 | 310.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 311.15 | 307.77 | 310.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 312.00 | 308.61 | 310.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 312.00 | 308.61 | 310.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 316.30 | 311.85 | 311.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 319.75 | 313.43 | 312.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 330.40 | 331.26 | 326.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 326.50 | 329.52 | 327.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 326.50 | 329.52 | 327.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 326.50 | 329.52 | 327.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 326.05 | 328.82 | 326.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 326.45 | 328.82 | 326.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 327.20 | 328.17 | 326.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 329.70 | 328.17 | 326.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 327.65 | 327.90 | 326.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 328.20 | 327.41 | 326.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-17 09:15:00 | 360.42 | 352.62 | 347.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 358.40 | 361.11 | 361.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 356.65 | 359.05 | 360.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 353.30 | 352.04 | 355.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 353.30 | 352.04 | 355.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 353.30 | 352.04 | 355.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 342.30 | 349.08 | 352.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 344.00 | 342.79 | 345.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 345.70 | 342.88 | 342.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 345.70 | 342.88 | 342.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 345.90 | 344.22 | 343.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 344.20 | 345.15 | 344.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 344.20 | 345.15 | 344.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 344.20 | 345.15 | 344.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 344.20 | 345.15 | 344.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 344.20 | 344.96 | 344.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 344.20 | 344.96 | 344.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 344.70 | 344.91 | 344.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 345.70 | 344.91 | 344.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 343.20 | 347.08 | 347.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 343.20 | 347.08 | 347.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 338.25 | 342.53 | 344.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 335.85 | 335.07 | 337.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 335.85 | 335.07 | 337.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 338.15 | 335.69 | 337.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 338.15 | 335.69 | 337.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 338.25 | 336.20 | 337.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 338.25 | 336.20 | 337.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 339.00 | 336.76 | 337.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 339.15 | 336.76 | 337.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 338.50 | 337.37 | 338.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 339.30 | 337.37 | 338.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 339.30 | 337.76 | 338.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 340.55 | 337.76 | 338.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 336.25 | 337.01 | 337.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 334.10 | 336.93 | 337.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 332.60 | 330.64 | 330.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 332.60 | 330.64 | 330.42 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 329.30 | 331.07 | 331.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 328.40 | 330.01 | 330.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 330.80 | 330.02 | 330.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 330.80 | 330.02 | 330.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 330.80 | 330.02 | 330.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 330.50 | 330.02 | 330.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 329.90 | 330.00 | 330.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 329.00 | 330.00 | 330.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 329.30 | 329.34 | 329.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:15:00 | 312.55 | 317.28 | 317.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 317.65 | 317.28 | 317.98 | SL hit (close>static) qty=0.50 sl=317.28 alert=retest2 |

### Cycle 23 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 320.60 | 317.61 | 317.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 332.60 | 322.03 | 319.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 321.55 | 325.07 | 322.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 321.55 | 325.07 | 322.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 321.55 | 325.07 | 322.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 321.55 | 325.07 | 322.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 322.75 | 324.61 | 322.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 323.35 | 324.35 | 322.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 14:15:00 | 321.45 | 323.30 | 322.73 | SL hit (close<static) qty=1.00 sl=321.60 alert=retest2 |

### Cycle 24 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 318.90 | 322.08 | 322.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 317.15 | 321.09 | 321.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 320.55 | 320.06 | 320.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 320.55 | 320.06 | 320.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 321.45 | 320.33 | 320.95 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 12:15:00 | 321.60 | 321.30 | 321.30 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 321.10 | 321.26 | 321.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 319.15 | 320.84 | 321.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 316.75 | 316.09 | 317.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 316.75 | 316.09 | 317.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 316.75 | 316.09 | 317.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 316.20 | 316.09 | 317.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 327.65 | 317.62 | 317.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 327.65 | 317.62 | 317.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 328.35 | 319.77 | 318.73 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 320.95 | 322.43 | 322.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 317.75 | 321.49 | 322.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 315.55 | 314.34 | 316.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 315.55 | 314.34 | 316.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 308.70 | 307.20 | 309.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 308.70 | 307.20 | 309.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 310.35 | 307.83 | 309.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 310.35 | 307.83 | 309.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 311.30 | 308.53 | 309.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 312.70 | 308.53 | 309.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 313.30 | 310.24 | 310.02 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 308.60 | 310.27 | 310.29 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 312.25 | 310.27 | 310.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 312.80 | 311.05 | 310.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 312.60 | 312.89 | 312.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 312.60 | 312.89 | 312.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 312.60 | 312.89 | 312.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 312.40 | 312.89 | 312.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 313.50 | 313.20 | 312.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 311.15 | 313.20 | 312.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 310.80 | 312.72 | 312.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 310.80 | 312.72 | 312.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 310.40 | 312.26 | 312.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 310.10 | 311.82 | 312.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 309.65 | 306.79 | 307.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 309.65 | 306.79 | 307.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 309.65 | 306.79 | 307.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 309.95 | 306.79 | 307.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 309.05 | 307.25 | 307.89 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 310.55 | 308.32 | 308.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 314.90 | 309.63 | 308.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 339.70 | 339.80 | 332.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 339.70 | 339.80 | 332.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 366.90 | 371.97 | 366.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 364.15 | 371.97 | 366.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 368.40 | 371.26 | 366.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 368.50 | 371.26 | 366.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 363.80 | 369.77 | 366.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 363.80 | 369.77 | 366.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 361.80 | 368.17 | 365.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:45:00 | 361.65 | 368.17 | 365.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 359.30 | 364.01 | 364.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 13:15:00 | 357.30 | 360.53 | 362.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 360.65 | 359.34 | 361.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 360.65 | 359.34 | 361.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 360.65 | 359.34 | 361.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 360.65 | 359.34 | 361.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 360.50 | 359.57 | 361.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 360.70 | 359.57 | 361.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 357.90 | 359.24 | 360.75 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 364.15 | 361.48 | 361.34 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 359.30 | 362.42 | 362.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 358.35 | 361.06 | 361.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 360.20 | 359.20 | 360.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 360.20 | 359.20 | 360.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 360.20 | 359.20 | 360.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 356.50 | 358.54 | 359.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 338.68 | 346.60 | 351.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 333.90 | 331.23 | 336.84 | SL hit (close>ema200) qty=0.50 sl=331.23 alert=retest2 |

### Cycle 37 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 338.10 | 334.99 | 334.61 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 332.00 | 335.21 | 335.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 330.90 | 334.35 | 335.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 322.30 | 319.45 | 322.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 322.30 | 319.45 | 322.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 322.30 | 319.45 | 322.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:00:00 | 320.65 | 319.69 | 322.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:45:00 | 321.00 | 320.17 | 322.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 329.10 | 321.95 | 322.85 | SL hit (close>static) qty=1.00 sl=326.00 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 330.20 | 323.60 | 323.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 333.45 | 325.57 | 324.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 327.25 | 327.52 | 325.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:15:00 | 327.20 | 327.52 | 325.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 325.45 | 327.10 | 325.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 325.45 | 327.10 | 325.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 325.00 | 326.68 | 325.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 326.90 | 326.68 | 325.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 324.55 | 326.39 | 325.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 324.55 | 326.39 | 325.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 324.25 | 325.96 | 325.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 324.25 | 325.96 | 325.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 322.10 | 325.19 | 325.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 321.85 | 324.52 | 325.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 325.90 | 324.66 | 324.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 325.90 | 324.66 | 324.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 325.90 | 324.66 | 324.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 330.15 | 324.66 | 324.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 326.95 | 325.12 | 325.17 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 330.85 | 326.27 | 325.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 334.50 | 327.91 | 326.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 332.90 | 335.57 | 331.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:30:00 | 335.90 | 335.57 | 331.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 333.30 | 335.11 | 331.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 331.75 | 335.11 | 331.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 338.95 | 337.50 | 334.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:45:00 | 340.15 | 338.04 | 334.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 339.50 | 338.46 | 335.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 340.65 | 339.51 | 336.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 329.95 | 338.87 | 337.76 | SL hit (close<static) qty=1.00 sl=331.80 alert=retest2 |

### Cycle 42 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 326.80 | 336.46 | 336.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 319.50 | 330.15 | 333.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 322.85 | 321.37 | 327.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 322.85 | 321.37 | 327.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 323.70 | 322.37 | 326.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 320.15 | 323.97 | 325.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 321.70 | 323.84 | 325.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 12:15:00 | 318.10 | 317.32 | 317.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 318.10 | 317.32 | 317.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 14:15:00 | 318.95 | 317.67 | 317.41 | Break + close above crossover candle high |

### Cycle 44 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 314.25 | 317.28 | 317.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 310.25 | 315.14 | 316.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 313.70 | 313.21 | 314.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 313.70 | 313.21 | 314.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 314.20 | 313.41 | 314.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 308.80 | 313.41 | 314.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 311.65 | 309.03 | 308.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 311.65 | 309.03 | 308.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 317.20 | 312.70 | 310.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 315.50 | 315.91 | 313.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:30:00 | 313.95 | 315.91 | 313.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 318.90 | 316.46 | 314.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:45:00 | 315.75 | 316.46 | 314.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 315.75 | 317.65 | 316.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 315.75 | 317.65 | 316.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 317.85 | 317.69 | 316.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:30:00 | 316.05 | 317.69 | 316.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 316.30 | 317.41 | 316.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 318.90 | 317.41 | 316.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 319.30 | 317.95 | 317.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 318.55 | 317.49 | 317.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 315.60 | 316.94 | 317.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 315.60 | 316.94 | 317.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 303.80 | 314.31 | 315.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 281.30 | 280.51 | 287.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 286.00 | 280.51 | 287.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 296.90 | 283.79 | 288.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 296.90 | 283.79 | 288.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 290.55 | 285.14 | 288.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 289.40 | 285.14 | 288.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 274.93 | 284.70 | 287.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 279.60 | 278.20 | 281.81 | SL hit (close>ema200) qty=0.50 sl=278.20 alert=retest2 |

### Cycle 47 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 275.00 | 270.80 | 270.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 275.80 | 271.80 | 270.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 266.85 | 271.81 | 271.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 266.85 | 271.81 | 271.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 266.85 | 271.81 | 271.12 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 267.30 | 270.09 | 270.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 265.70 | 269.21 | 269.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 267.85 | 266.57 | 268.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 267.85 | 266.57 | 268.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 267.85 | 266.57 | 268.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 268.40 | 266.57 | 268.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 266.80 | 266.62 | 268.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 267.90 | 266.62 | 268.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 265.35 | 266.36 | 267.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 264.35 | 266.36 | 267.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 264.60 | 265.89 | 267.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 251.37 | 261.10 | 264.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 251.13 | 257.28 | 262.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 256.70 | 254.55 | 259.04 | SL hit (close>ema200) qty=0.50 sl=254.55 alert=retest2 |

### Cycle 49 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 272.00 | 262.02 | 260.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 272.70 | 264.16 | 261.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 264.10 | 267.02 | 264.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 264.10 | 267.02 | 264.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 264.10 | 267.02 | 264.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 264.10 | 267.02 | 264.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 262.65 | 266.15 | 264.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 262.25 | 266.15 | 264.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 263.90 | 265.70 | 264.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 262.70 | 265.70 | 264.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 266.00 | 265.77 | 264.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:30:00 | 265.75 | 265.77 | 264.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 264.00 | 265.41 | 264.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 264.00 | 265.41 | 264.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 264.05 | 265.14 | 264.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 258.20 | 265.14 | 264.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 258.20 | 263.75 | 263.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 255.90 | 262.18 | 263.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 262.90 | 256.68 | 259.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 262.90 | 256.68 | 259.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 262.90 | 256.68 | 259.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 264.81 | 256.68 | 259.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 259.70 | 257.28 | 259.35 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 262.07 | 260.74 | 260.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 264.95 | 261.58 | 260.99 | Break + close above crossover candle high |

### Cycle 52 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 252.38 | 259.74 | 260.20 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 261.35 | 259.08 | 258.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 262.90 | 260.25 | 259.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 259.09 | 260.02 | 259.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 259.09 | 260.02 | 259.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 259.09 | 260.02 | 259.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 271.33 | 260.97 | 260.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 11:15:00 | 298.46 | 287.39 | 281.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 303.64 | 306.06 | 306.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 302.89 | 305.43 | 305.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 300.30 | 298.78 | 301.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 300.30 | 298.78 | 301.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 300.30 | 298.78 | 301.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 298.60 | 299.04 | 300.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 298.70 | 298.26 | 299.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 304.40 | 299.64 | 299.86 | SL hit (close>static) qty=1.00 sl=303.40 alert=retest2 |

### Cycle 55 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 301.00 | 300.06 | 300.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 302.15 | 300.48 | 300.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 300.25 | 300.75 | 300.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 12:15:00 | 300.25 | 300.75 | 300.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 300.25 | 300.75 | 300.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 300.25 | 300.75 | 300.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 301.80 | 300.96 | 300.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 303.65 | 300.96 | 300.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 10:00:00 | 417.15 | 2025-05-21 11:15:00 | 409.85 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-05-26 11:30:00 | 407.45 | 2025-05-26 13:15:00 | 409.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-05-29 15:15:00 | 419.30 | 2025-05-30 10:15:00 | 413.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-03 13:00:00 | 403.35 | 2025-06-04 10:15:00 | 417.30 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-06-03 13:30:00 | 403.90 | 2025-06-04 10:15:00 | 417.30 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-06-03 14:00:00 | 403.90 | 2025-06-04 10:15:00 | 417.30 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-06-03 15:15:00 | 403.90 | 2025-06-04 10:15:00 | 417.30 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-06-06 14:00:00 | 429.95 | 2025-06-12 10:15:00 | 423.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-10 11:45:00 | 426.85 | 2025-06-12 10:15:00 | 423.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-10 15:00:00 | 427.90 | 2025-06-12 10:15:00 | 423.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-11 09:15:00 | 428.35 | 2025-06-12 10:15:00 | 423.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-17 11:45:00 | 405.35 | 2025-06-19 12:15:00 | 385.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 405.35 | 2025-06-20 10:15:00 | 388.00 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2025-07-03 12:30:00 | 392.35 | 2025-07-22 14:15:00 | 372.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 13:15:00 | 392.15 | 2025-07-22 14:15:00 | 372.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 13:45:00 | 391.75 | 2025-07-22 14:15:00 | 372.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-04 09:30:00 | 392.25 | 2025-07-22 14:15:00 | 372.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:45:00 | 388.60 | 2025-07-23 09:15:00 | 369.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 12:30:00 | 388.20 | 2025-07-23 10:15:00 | 368.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 09:30:00 | 387.45 | 2025-07-23 11:15:00 | 368.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 12:30:00 | 392.35 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest2 | 2025-07-03 13:15:00 | 392.15 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-07-03 13:45:00 | 391.75 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2025-07-04 09:30:00 | 392.25 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2025-07-08 09:45:00 | 388.60 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-07-09 12:30:00 | 388.20 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2025-07-10 09:30:00 | 387.45 | 2025-07-23 12:15:00 | 372.90 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-07-30 10:15:00 | 357.15 | 2025-08-05 10:15:00 | 354.65 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-07-31 09:15:00 | 354.00 | 2025-08-05 10:15:00 | 354.65 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-08-08 09:15:00 | 345.15 | 2025-08-12 11:15:00 | 344.70 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-08-14 10:30:00 | 326.95 | 2025-08-20 10:15:00 | 331.15 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-18 10:45:00 | 326.00 | 2025-08-20 10:15:00 | 331.15 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-08-20 09:15:00 | 324.95 | 2025-08-20 10:15:00 | 331.15 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-05 09:15:00 | 329.70 | 2025-09-17 09:15:00 | 360.42 | TARGET_HIT | 1.00 | 9.32% |
| BUY | retest2 | 2025-09-05 10:15:00 | 327.65 | 2025-09-17 09:15:00 | 361.02 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2025-09-05 13:30:00 | 328.20 | 2025-09-17 10:15:00 | 362.67 | TARGET_HIT | 1.00 | 10.50% |
| SELL | retest2 | 2025-09-26 09:15:00 | 342.30 | 2025-10-01 15:15:00 | 345.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-29 11:30:00 | 344.00 | 2025-10-01 15:15:00 | 345.70 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-06 13:15:00 | 345.70 | 2025-10-09 09:15:00 | 343.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-17 09:15:00 | 334.10 | 2025-10-29 12:15:00 | 332.60 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-11-03 11:15:00 | 329.00 | 2025-11-12 09:15:00 | 312.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 11:15:00 | 329.00 | 2025-11-12 09:15:00 | 317.65 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-11-03 14:45:00 | 329.30 | 2025-11-12 09:15:00 | 312.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:45:00 | 329.30 | 2025-11-12 09:15:00 | 317.65 | STOP_HIT | 0.50 | 3.54% |
| BUY | retest2 | 2025-11-18 12:00:00 | 323.35 | 2025-11-18 14:15:00 | 321.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-08 09:15:00 | 356.50 | 2026-01-09 09:15:00 | 338.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 356.50 | 2026-01-12 15:15:00 | 333.90 | STOP_HIT | 0.50 | 6.34% |
| SELL | retest2 | 2026-01-22 11:00:00 | 320.65 | 2026-01-22 14:15:00 | 329.10 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-01-22 13:45:00 | 321.00 | 2026-01-22 14:15:00 | 329.10 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-01-30 10:45:00 | 340.15 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2026-01-30 13:30:00 | 339.50 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-01-30 14:45:00 | 340.65 | 2026-02-01 12:15:00 | 329.95 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-02-04 09:15:00 | 320.15 | 2026-02-10 12:15:00 | 318.10 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-02-04 10:15:00 | 321.70 | 2026-02-10 12:15:00 | 318.10 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2026-02-13 09:15:00 | 308.80 | 2026-02-20 12:15:00 | 311.65 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-26 09:15:00 | 318.90 | 2026-02-27 15:15:00 | 315.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-26 14:30:00 | 319.30 | 2026-02-27 15:15:00 | 315.60 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-27 13:45:00 | 318.55 | 2026-02-27 15:15:00 | 315.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-06 11:15:00 | 289.40 | 2026-03-09 09:15:00 | 274.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 11:15:00 | 289.40 | 2026-03-10 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2026-03-20 12:15:00 | 264.35 | 2026-03-23 10:15:00 | 251.37 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-03-20 14:15:00 | 264.60 | 2026-03-23 12:15:00 | 251.13 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2026-03-20 12:15:00 | 264.35 | 2026-03-24 09:15:00 | 256.70 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-03-20 14:15:00 | 264.60 | 2026-03-24 09:15:00 | 256.70 | STOP_HIT | 0.50 | 2.99% |
| BUY | retest2 | 2026-04-08 09:15:00 | 271.33 | 2026-04-16 11:15:00 | 298.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 12:00:00 | 298.60 | 2026-05-05 12:15:00 | 304.40 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-05-05 10:30:00 | 298.70 | 2026-05-05 12:15:00 | 304.40 | STOP_HIT | 1.00 | -1.91% |
