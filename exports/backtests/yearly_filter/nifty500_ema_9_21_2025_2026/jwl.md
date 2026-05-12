# Jupiter Wagons Ltd. (JWL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 298.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 48 |
| ALERT2 | 46 |
| ALERT2_SKIP | 17 |
| ALERT3 | 110 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 20 |
| TARGET_HIT | 7 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 13
- **Target hits / Stop hits / Partials:** 7 / 32 / 20
- **Avg / median % per leg:** 3.27% / 4.77%
- **Sum % (uncompounded):** 193.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.24% | -8.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.24% | -8.7% |
| SELL (all) | 52 | 46 | 88.5% | 7 | 25 | 20 | 3.88% | 201.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 46 | 88.5% | 7 | 25 | 20 | 3.88% | 201.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 59 | 46 | 78.0% | 7 | 32 | 20 | 3.27% | 193.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 360.95 | 344.64 | 343.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 367.00 | 354.07 | 348.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 410.35 | 429.45 | 415.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 14:15:00 | 410.35 | 429.45 | 415.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 410.35 | 429.45 | 415.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 410.35 | 429.45 | 415.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 408.95 | 425.35 | 415.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 395.50 | 425.35 | 415.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 398.85 | 415.36 | 412.10 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 395.85 | 408.90 | 409.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 394.65 | 406.05 | 408.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 403.80 | 401.96 | 405.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 403.80 | 401.96 | 405.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 400.25 | 398.93 | 401.70 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 402.90 | 399.61 | 399.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 411.10 | 402.97 | 401.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 406.20 | 407.22 | 404.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:15:00 | 401.80 | 407.22 | 404.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 403.45 | 406.47 | 404.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 403.30 | 406.47 | 404.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 403.70 | 405.91 | 404.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 402.95 | 405.91 | 404.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 404.40 | 405.91 | 405.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 404.40 | 405.91 | 405.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 405.30 | 405.79 | 405.04 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 398.65 | 404.44 | 404.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 397.10 | 400.61 | 402.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 399.75 | 399.05 | 401.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 399.75 | 399.05 | 401.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 399.75 | 399.05 | 401.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 399.80 | 399.05 | 401.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 402.50 | 399.69 | 400.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 396.25 | 399.25 | 400.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 11:15:00 | 412.80 | 398.10 | 397.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 412.80 | 398.10 | 397.84 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 399.95 | 402.29 | 402.44 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 408.10 | 403.43 | 402.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 409.90 | 406.62 | 404.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 408.00 | 408.35 | 406.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 407.25 | 408.35 | 406.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 410.65 | 408.81 | 407.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 407.50 | 408.81 | 407.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 405.50 | 408.78 | 407.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 404.55 | 408.78 | 407.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 409.10 | 408.84 | 407.76 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 403.10 | 406.60 | 406.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 393.20 | 403.43 | 405.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 389.30 | 389.23 | 393.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 389.30 | 389.23 | 393.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 389.75 | 389.95 | 392.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 392.00 | 389.95 | 392.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 386.60 | 387.04 | 389.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 384.25 | 386.93 | 389.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 384.45 | 386.93 | 389.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 365.04 | 376.07 | 381.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 365.23 | 376.07 | 381.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 373.35 | 371.34 | 376.74 | SL hit (close>ema200) qty=0.50 sl=371.34 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 383.30 | 375.88 | 375.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 388.40 | 384.99 | 382.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 387.45 | 387.72 | 385.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 387.45 | 387.72 | 385.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 385.05 | 387.23 | 386.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 385.85 | 387.23 | 386.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 384.20 | 386.62 | 386.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 384.20 | 386.62 | 386.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 384.55 | 385.64 | 385.73 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 387.50 | 386.10 | 385.93 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 384.00 | 385.68 | 385.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 383.35 | 384.65 | 385.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 383.25 | 383.24 | 383.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:45:00 | 383.15 | 383.24 | 383.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 381.40 | 381.66 | 382.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:30:00 | 379.70 | 380.78 | 381.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 378.85 | 376.42 | 377.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 378.75 | 376.92 | 377.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 379.00 | 376.92 | 377.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 376.70 | 376.88 | 377.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 375.75 | 376.65 | 377.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 372.50 | 371.41 | 371.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 372.50 | 371.41 | 371.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 373.90 | 371.91 | 371.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 371.80 | 372.09 | 371.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 371.80 | 372.09 | 371.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 371.80 | 372.09 | 371.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 372.15 | 372.09 | 371.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 371.75 | 372.02 | 371.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 371.75 | 372.02 | 371.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 370.40 | 371.70 | 371.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 370.40 | 371.70 | 371.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 370.30 | 371.42 | 371.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 369.95 | 370.98 | 371.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 367.95 | 367.75 | 369.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 367.50 | 367.70 | 368.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 371.25 | 368.41 | 369.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 371.25 | 368.41 | 369.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 369.75 | 368.68 | 369.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 369.30 | 368.92 | 369.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 369.35 | 369.17 | 369.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 369.00 | 369.17 | 369.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 350.83 | 354.52 | 358.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 350.88 | 354.52 | 358.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 350.55 | 354.52 | 358.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 346.15 | 345.01 | 349.32 | SL hit (close>ema200) qty=0.50 sl=345.01 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 345.15 | 342.52 | 342.24 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 337.55 | 342.01 | 342.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 333.25 | 337.73 | 339.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 338.35 | 335.08 | 337.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 338.35 | 335.08 | 337.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 338.35 | 335.08 | 337.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 338.35 | 335.08 | 337.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 339.00 | 335.87 | 337.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 337.25 | 335.72 | 337.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 320.39 | 328.41 | 331.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 325.10 | 317.45 | 320.35 | SL hit (close>ema200) qty=0.50 sl=317.45 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 324.80 | 321.76 | 321.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 326.25 | 322.66 | 322.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 327.90 | 328.53 | 326.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 327.90 | 328.53 | 326.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 341.40 | 348.11 | 340.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 338.30 | 348.11 | 340.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 340.65 | 346.62 | 340.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 341.05 | 346.62 | 340.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 340.10 | 345.32 | 340.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 340.10 | 345.32 | 340.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 339.65 | 344.18 | 340.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 339.65 | 344.18 | 340.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 340.00 | 343.35 | 340.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 339.25 | 343.35 | 340.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 340.65 | 342.24 | 340.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 339.50 | 342.24 | 340.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 340.65 | 341.92 | 340.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 335.50 | 341.92 | 340.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 336.20 | 340.78 | 340.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 336.30 | 340.78 | 340.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 336.75 | 339.97 | 339.87 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 336.75 | 339.33 | 339.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 336.40 | 338.74 | 339.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 322.75 | 322.74 | 327.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 322.40 | 322.74 | 327.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 323.40 | 321.01 | 322.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 324.75 | 321.01 | 322.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 327.75 | 322.36 | 323.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 328.40 | 322.36 | 323.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 330.40 | 323.97 | 323.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 335.25 | 327.35 | 325.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 330.20 | 330.23 | 328.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:15:00 | 328.60 | 330.23 | 328.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 329.10 | 330.01 | 328.64 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 326.10 | 327.95 | 327.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 325.50 | 327.46 | 327.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 324.80 | 324.36 | 325.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 326.80 | 324.36 | 325.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 332.20 | 325.93 | 326.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 331.50 | 325.93 | 326.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 329.20 | 326.58 | 326.47 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 325.00 | 326.36 | 326.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 323.45 | 325.63 | 326.13 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 334.60 | 323.20 | 323.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 346.00 | 335.25 | 331.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 341.65 | 342.31 | 337.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 10:15:00 | 340.85 | 342.31 | 337.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 342.60 | 344.11 | 343.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 342.60 | 344.11 | 343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 342.80 | 343.85 | 343.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 343.70 | 343.85 | 343.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 343.80 | 343.84 | 343.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:00:00 | 343.75 | 343.82 | 343.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 339.80 | 342.93 | 342.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 339.80 | 342.93 | 342.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 337.75 | 340.92 | 341.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 334.45 | 327.41 | 329.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 334.45 | 327.41 | 329.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 334.45 | 327.41 | 329.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 334.45 | 327.41 | 329.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 341.40 | 330.21 | 330.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 344.30 | 330.21 | 330.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 11:15:00 | 334.80 | 331.13 | 330.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 341.70 | 339.43 | 338.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 340.80 | 341.13 | 339.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:45:00 | 342.40 | 341.13 | 339.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 338.15 | 340.54 | 339.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 338.15 | 340.54 | 339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 339.60 | 340.35 | 339.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:30:00 | 339.90 | 340.37 | 339.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 336.00 | 339.24 | 339.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 336.00 | 339.24 | 339.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 335.00 | 337.97 | 338.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 335.85 | 335.72 | 337.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 335.85 | 335.72 | 337.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 336.05 | 335.90 | 336.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 331.50 | 335.11 | 336.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 323.70 | 322.82 | 322.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 323.70 | 322.82 | 322.73 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 321.35 | 322.61 | 322.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 320.85 | 321.92 | 322.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 323.00 | 321.99 | 322.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 323.00 | 321.99 | 322.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 323.00 | 321.99 | 322.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 323.60 | 321.99 | 322.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 322.80 | 322.15 | 322.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 324.80 | 322.15 | 322.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 323.30 | 322.38 | 322.37 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 321.50 | 322.22 | 322.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 320.85 | 321.95 | 322.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 321.60 | 321.32 | 321.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 321.60 | 321.32 | 321.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 321.60 | 321.32 | 321.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 323.75 | 321.32 | 321.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 323.85 | 321.83 | 321.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 323.85 | 321.83 | 321.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 325.65 | 322.59 | 322.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 327.25 | 324.03 | 323.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 326.50 | 326.88 | 325.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 326.50 | 326.88 | 325.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 326.50 | 326.81 | 325.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 325.85 | 326.81 | 325.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 324.80 | 326.21 | 325.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 324.80 | 326.21 | 325.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 322.80 | 325.53 | 325.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 322.80 | 325.53 | 325.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 323.45 | 324.88 | 324.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 322.70 | 324.23 | 324.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 324.75 | 324.06 | 324.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 324.75 | 324.06 | 324.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 324.75 | 324.06 | 324.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 322.30 | 323.37 | 323.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 14:15:00 | 306.19 | 313.10 | 313.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 313.75 | 313.10 | 313.98 | SL hit (close>static) qty=0.50 sl=313.10 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 310.40 | 308.16 | 308.15 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 305.05 | 308.11 | 308.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 303.55 | 305.55 | 306.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 303.55 | 303.32 | 304.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 303.55 | 303.32 | 304.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 303.55 | 303.32 | 304.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 304.65 | 303.32 | 304.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 291.05 | 290.27 | 292.69 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 293.40 | 292.81 | 292.80 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 292.00 | 292.65 | 292.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 289.10 | 291.80 | 292.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 289.95 | 288.82 | 290.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 289.95 | 288.82 | 290.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 289.95 | 288.82 | 290.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 289.25 | 288.82 | 290.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 287.10 | 288.47 | 289.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:15:00 | 286.50 | 288.47 | 289.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 272.18 | 275.54 | 278.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 10:15:00 | 257.85 | 266.43 | 271.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 261.05 | 258.40 | 258.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 261.85 | 260.11 | 259.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 263.65 | 265.32 | 262.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:30:00 | 264.85 | 265.32 | 262.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 263.55 | 264.60 | 262.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 262.50 | 264.60 | 262.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 263.35 | 264.35 | 263.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 263.70 | 264.35 | 263.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 263.15 | 264.11 | 263.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:15:00 | 262.60 | 264.11 | 263.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 262.80 | 263.85 | 263.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 262.75 | 263.85 | 263.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 262.75 | 263.63 | 262.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 261.80 | 263.63 | 262.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 259.70 | 262.84 | 262.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 259.70 | 262.84 | 262.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 259.35 | 262.14 | 262.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 258.35 | 260.92 | 261.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 255.85 | 255.71 | 257.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 255.85 | 255.71 | 257.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 255.85 | 255.71 | 257.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 256.85 | 255.71 | 257.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 255.65 | 255.70 | 257.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 257.30 | 255.70 | 257.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 259.60 | 256.14 | 257.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 258.85 | 256.14 | 257.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 260.15 | 256.94 | 257.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 286.20 | 256.94 | 257.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 290.90 | 263.73 | 260.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 292.05 | 269.40 | 263.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 345.20 | 345.64 | 335.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 342.55 | 345.64 | 335.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 338.80 | 341.87 | 335.98 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 332.60 | 334.64 | 334.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 331.10 | 333.93 | 334.48 | Break + close below crossover candle low |

### Cycle 41 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 347.30 | 336.10 | 335.34 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 336.85 | 337.38 | 337.41 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 338.15 | 337.38 | 337.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 339.95 | 338.20 | 337.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 337.85 | 338.13 | 337.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 337.85 | 338.13 | 337.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 337.85 | 338.13 | 337.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 337.85 | 338.13 | 337.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 339.00 | 338.30 | 337.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 337.50 | 338.30 | 337.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 333.35 | 337.31 | 337.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 330.55 | 335.96 | 336.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 295.20 | 294.42 | 302.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 294.55 | 294.42 | 302.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 296.65 | 295.85 | 300.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 292.40 | 294.77 | 298.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 326.20 | 301.43 | 300.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 326.20 | 301.43 | 300.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 330.90 | 311.15 | 304.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 319.90 | 321.52 | 315.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 323.00 | 321.52 | 315.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 326.40 | 322.50 | 316.87 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 315.50 | 316.86 | 317.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 309.95 | 315.48 | 316.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 315.70 | 313.88 | 315.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 315.70 | 313.88 | 315.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 315.70 | 313.88 | 315.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:30:00 | 315.00 | 313.88 | 315.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 311.30 | 313.36 | 314.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 305.00 | 309.38 | 311.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:45:00 | 303.55 | 307.79 | 309.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:45:00 | 305.15 | 307.19 | 309.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 314.15 | 309.82 | 309.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 314.15 | 309.82 | 309.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 321.10 | 312.08 | 310.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 324.65 | 324.72 | 320.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 324.65 | 324.72 | 320.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 321.50 | 325.96 | 323.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 321.50 | 325.96 | 323.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 315.35 | 323.84 | 322.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 314.55 | 323.84 | 322.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 310.30 | 321.13 | 321.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 301.50 | 317.21 | 319.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 308.25 | 307.02 | 312.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 308.25 | 307.02 | 312.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 309.30 | 307.64 | 311.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 307.30 | 309.53 | 310.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 309.05 | 304.04 | 303.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 309.05 | 304.04 | 303.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 313.70 | 306.71 | 304.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 310.00 | 310.71 | 308.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 307.90 | 310.71 | 308.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 307.40 | 310.05 | 308.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 306.45 | 310.05 | 308.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 307.50 | 309.54 | 307.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 307.50 | 309.54 | 307.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 307.65 | 309.16 | 307.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 307.65 | 309.16 | 307.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 308.50 | 309.03 | 307.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 307.35 | 309.03 | 307.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 308.50 | 308.92 | 308.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 308.50 | 308.92 | 308.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 310.95 | 309.33 | 308.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:15:00 | 308.00 | 309.33 | 308.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 308.00 | 309.06 | 308.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 297.70 | 309.06 | 308.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 299.90 | 307.23 | 307.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 293.80 | 300.00 | 303.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 292.80 | 291.26 | 294.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 293.85 | 291.26 | 294.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 293.75 | 291.76 | 294.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 289.05 | 290.81 | 292.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 288.50 | 291.44 | 291.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:45:00 | 288.90 | 289.82 | 290.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 289.25 | 289.28 | 290.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 287.95 | 289.07 | 289.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 287.00 | 289.07 | 289.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 287.25 | 288.16 | 289.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 287.30 | 288.01 | 289.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 285.70 | 288.64 | 289.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 285.40 | 287.99 | 288.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:30:00 | 284.30 | 286.73 | 288.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 283.40 | 285.44 | 287.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.60 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.07 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.45 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 15:15:00 | 274.79 | 280.19 | 283.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 280.95 | 280.34 | 282.92 | SL hit (close>ema200) qty=0.50 sl=280.34 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 286.60 | 259.30 | 258.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 10:15:00 | 287.95 | 265.03 | 260.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 13:15:00 | 285.70 | 288.39 | 280.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:00:00 | 285.70 | 288.39 | 280.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 280.55 | 286.22 | 281.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 277.75 | 286.22 | 281.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 283.75 | 285.73 | 281.42 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 278.90 | 281.79 | 281.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 277.15 | 280.86 | 281.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 262.55 | 261.68 | 266.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 264.85 | 261.68 | 266.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 263.25 | 261.88 | 265.37 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 278.50 | 266.21 | 265.80 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 263.25 | 267.08 | 267.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 262.30 | 266.12 | 267.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 265.45 | 265.41 | 266.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 265.45 | 265.41 | 266.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 265.45 | 265.41 | 266.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 263.30 | 265.28 | 266.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 263.95 | 264.50 | 265.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 250.75 | 259.80 | 263.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 250.13 | 256.37 | 260.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 256.20 | 255.08 | 258.02 | SL hit (close>ema200) qty=0.50 sl=255.08 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 265.90 | 259.47 | 259.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 268.45 | 261.27 | 260.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 261.70 | 262.70 | 261.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 261.70 | 262.70 | 261.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 261.60 | 262.48 | 261.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 259.00 | 262.48 | 261.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 254.30 | 260.85 | 260.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 254.30 | 260.85 | 260.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 252.25 | 259.13 | 259.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 250.80 | 255.26 | 257.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 253.50 | 245.95 | 250.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 253.50 | 245.95 | 250.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 253.50 | 245.95 | 250.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 255.90 | 245.95 | 250.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 251.93 | 247.15 | 250.27 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 255.75 | 252.33 | 252.06 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 248.10 | 251.42 | 251.76 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 256.98 | 252.83 | 252.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 258.03 | 255.42 | 253.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 266.25 | 266.62 | 261.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 266.25 | 266.62 | 261.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 269.15 | 270.87 | 268.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 269.22 | 270.87 | 268.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 269.52 | 270.60 | 268.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 269.52 | 270.60 | 268.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 269.50 | 270.38 | 268.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 272.94 | 270.38 | 268.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 268.00 | 269.63 | 269.24 | SL hit (close<static) qty=1.00 sl=268.08 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 260.00 | 267.70 | 268.40 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 273.85 | 267.30 | 267.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 282.80 | 273.86 | 270.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 281.84 | 284.20 | 280.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 281.84 | 284.20 | 280.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 281.84 | 284.20 | 280.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 279.30 | 284.20 | 280.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 280.55 | 282.81 | 280.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 280.61 | 282.81 | 280.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 279.86 | 282.22 | 280.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 279.86 | 282.22 | 280.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 279.00 | 281.58 | 280.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 281.44 | 281.58 | 280.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:00:00 | 282.00 | 285.13 | 284.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 278.55 | 283.81 | 283.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 278.55 | 283.81 | 283.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 277.65 | 281.85 | 282.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 282.81 | 280.99 | 282.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 282.81 | 280.99 | 282.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 282.81 | 280.99 | 282.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 283.99 | 280.99 | 282.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 286.80 | 282.15 | 282.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 287.02 | 282.15 | 282.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 285.76 | 282.87 | 282.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 288.05 | 283.91 | 283.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 289.10 | 289.22 | 287.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 289.10 | 289.22 | 287.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 287.62 | 289.35 | 288.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 287.62 | 289.35 | 288.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 287.40 | 288.96 | 288.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 287.40 | 288.96 | 288.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 287.50 | 288.67 | 288.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 283.85 | 288.67 | 288.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 281.39 | 287.21 | 287.56 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 287.75 | 286.35 | 286.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 289.70 | 287.02 | 286.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 294.65 | 295.15 | 292.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:45:00 | 294.40 | 295.15 | 292.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 298.25 | 300.53 | 298.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 298.25 | 300.53 | 298.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 298.90 | 300.20 | 298.79 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-03 09:45:00 | 396.25 | 2025-06-04 11:15:00 | 412.80 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-06-18 11:45:00 | 384.25 | 2025-06-19 12:15:00 | 365.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 384.45 | 2025-06-19 12:15:00 | 365.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 11:45:00 | 384.25 | 2025-06-20 10:15:00 | 373.35 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2025-06-18 12:15:00 | 384.45 | 2025-06-20 10:15:00 | 373.35 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-07-07 12:30:00 | 379.70 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.90% |
| SELL | retest2 | 2025-07-09 11:30:00 | 378.85 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-07-10 09:30:00 | 378.75 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-07-10 10:15:00 | 379.00 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2025-07-10 12:00:00 | 375.75 | 2025-07-16 12:15:00 | 372.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-07-21 13:30:00 | 369.30 | 2025-07-28 09:15:00 | 350.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:30:00 | 369.35 | 2025-07-28 09:15:00 | 350.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 369.00 | 2025-07-28 09:15:00 | 350.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:30:00 | 369.30 | 2025-07-29 12:15:00 | 346.15 | STOP_HIT | 0.50 | 6.27% |
| SELL | retest2 | 2025-07-22 09:30:00 | 369.35 | 2025-07-29 12:15:00 | 346.15 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2025-07-22 10:15:00 | 369.00 | 2025-07-29 12:15:00 | 346.15 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2025-08-08 09:30:00 | 337.25 | 2025-08-13 09:15:00 | 320.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:30:00 | 337.25 | 2025-08-18 09:15:00 | 325.10 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2025-09-19 11:15:00 | 343.70 | 2025-09-19 14:15:00 | 339.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-19 12:00:00 | 343.80 | 2025-09-19 14:15:00 | 339.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-19 13:00:00 | 343.75 | 2025-09-19 14:15:00 | 339.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-08 12:30:00 | 339.90 | 2025-10-08 14:15:00 | 336.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-13 09:15:00 | 331.50 | 2025-10-23 09:15:00 | 323.70 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-11-04 11:15:00 | 322.30 | 2025-11-11 14:15:00 | 306.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 322.30 | 2025-11-11 14:15:00 | 313.75 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-12-01 11:15:00 | 286.50 | 2025-12-05 09:15:00 | 272.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:15:00 | 286.50 | 2025-12-08 10:15:00 | 257.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 292.40 | 2026-01-14 11:15:00 | 326.20 | STOP_HIT | 1.00 | -11.56% |
| SELL | retest2 | 2026-01-23 15:00:00 | 305.00 | 2026-01-28 10:15:00 | 314.15 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-01-27 10:45:00 | 303.55 | 2026-01-28 10:15:00 | 314.15 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-01-27 13:45:00 | 305.15 | 2026-01-28 10:15:00 | 314.15 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2026-02-04 09:15:00 | 307.30 | 2026-02-09 14:15:00 | 309.05 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-18 09:30:00 | 289.05 | 2026-02-25 15:15:00 | 274.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 15:15:00 | 288.50 | 2026-02-25 15:15:00 | 274.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 10:45:00 | 288.90 | 2026-02-25 15:15:00 | 274.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 15:00:00 | 289.25 | 2026-02-25 15:15:00 | 274.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:30:00 | 289.05 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-02-19 15:15:00 | 288.50 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2026-02-20 10:45:00 | 288.90 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2026-02-20 15:00:00 | 289.25 | 2026-02-26 09:15:00 | 280.95 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2026-02-23 10:15:00 | 287.00 | 2026-02-27 09:15:00 | 272.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 287.25 | 2026-02-27 09:15:00 | 272.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:30:00 | 287.30 | 2026-02-27 09:15:00 | 272.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 285.70 | 2026-02-27 09:15:00 | 271.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:30:00 | 284.30 | 2026-02-27 09:15:00 | 270.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 15:15:00 | 283.40 | 2026-02-27 09:15:00 | 269.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 287.00 | 2026-03-02 09:15:00 | 258.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 287.25 | 2026-03-02 09:15:00 | 258.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 13:30:00 | 287.30 | 2026-03-02 09:15:00 | 258.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 285.70 | 2026-03-02 09:15:00 | 257.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 11:30:00 | 284.30 | 2026-03-02 09:15:00 | 255.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 15:15:00 | 283.40 | 2026-03-02 09:15:00 | 255.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 263.30 | 2026-03-23 10:15:00 | 250.75 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-03-20 13:30:00 | 263.95 | 2026-03-23 12:15:00 | 250.13 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2026-03-20 12:15:00 | 263.30 | 2026-03-24 11:15:00 | 256.20 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2026-03-20 13:30:00 | 263.95 | 2026-03-24 11:15:00 | 256.20 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2026-04-10 09:15:00 | 272.94 | 2026-04-10 15:15:00 | 268.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-21 09:15:00 | 281.44 | 2026-04-24 10:15:00 | 278.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-04-24 10:00:00 | 282.00 | 2026-04-24 10:15:00 | 278.55 | STOP_HIT | 1.00 | -1.22% |
