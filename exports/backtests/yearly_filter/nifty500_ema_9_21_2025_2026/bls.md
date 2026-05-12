# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 290.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 50 |
| ALERT2 | 48 |
| ALERT2_SKIP | 23 |
| ALERT3 | 100 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 47 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 30
- **Target hits / Stop hits / Partials:** 4 / 47 / 9
- **Avg / median % per leg:** 0.64% / 0.10%
- **Sum % (uncompounded):** 38.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 3 | 17.6% | 2 | 15 | 0 | -0.33% | -5.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 3 | 17.6% | 2 | 15 | 0 | -0.33% | -5.7% |
| SELL (all) | 43 | 27 | 62.8% | 2 | 32 | 9 | 1.03% | 44.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.44% | 1.7% |
| SELL @ 3rd Alert (retest2) | 39 | 23 | 59.0% | 2 | 28 | 9 | 1.09% | 42.5% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.44% | 1.7% |
| retest2 (combined) | 56 | 26 | 46.4% | 4 | 43 | 9 | 0.66% | 36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 360.15 | 344.70 | 344.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 364.80 | 358.58 | 352.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 388.20 | 388.44 | 380.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 388.20 | 388.44 | 380.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 392.35 | 397.54 | 395.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 392.35 | 397.54 | 395.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 394.00 | 396.84 | 394.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 395.65 | 396.84 | 394.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 396.20 | 397.32 | 396.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 395.00 | 396.76 | 396.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 395.00 | 396.41 | 396.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 395.00 | 396.41 | 396.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 393.95 | 395.92 | 396.20 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 399.05 | 396.33 | 396.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 401.70 | 397.40 | 396.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 409.85 | 411.10 | 406.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:30:00 | 411.05 | 411.10 | 406.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 419.40 | 412.49 | 409.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 414.90 | 412.49 | 409.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 413.20 | 413.89 | 411.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 412.70 | 413.89 | 411.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 411.75 | 413.46 | 411.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 412.25 | 413.46 | 411.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 410.70 | 412.91 | 411.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 410.70 | 412.91 | 411.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 411.75 | 412.68 | 411.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 411.75 | 412.68 | 411.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 409.80 | 412.10 | 411.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 409.80 | 412.10 | 411.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 408.80 | 411.44 | 411.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 408.45 | 411.44 | 411.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 408.90 | 410.93 | 411.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 406.15 | 408.34 | 409.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 413.20 | 409.07 | 409.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 413.20 | 409.07 | 409.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 413.20 | 409.07 | 409.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 413.20 | 409.07 | 409.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 412.00 | 409.65 | 409.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:15:00 | 413.70 | 409.65 | 409.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 416.40 | 411.00 | 410.47 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 406.50 | 410.17 | 410.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 14:15:00 | 406.20 | 409.37 | 410.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 409.25 | 408.35 | 409.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 409.25 | 408.35 | 409.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 409.25 | 408.35 | 409.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 409.25 | 408.35 | 409.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 410.50 | 408.78 | 409.56 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 411.65 | 410.08 | 409.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 412.60 | 410.58 | 410.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 409.35 | 410.42 | 410.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 14:15:00 | 409.35 | 410.42 | 410.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 409.35 | 410.42 | 410.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 409.35 | 410.42 | 410.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 407.70 | 409.88 | 409.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 406.60 | 409.22 | 409.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 374.45 | 373.96 | 378.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 374.45 | 373.96 | 378.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 370.00 | 368.38 | 371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 371.45 | 368.80 | 371.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 368.80 | 368.80 | 371.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 367.25 | 368.80 | 371.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 365.90 | 359.65 | 359.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 365.90 | 359.65 | 359.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 378.75 | 370.00 | 367.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 370.55 | 374.83 | 371.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 370.55 | 374.83 | 371.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 370.55 | 374.83 | 371.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 370.55 | 374.83 | 371.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 370.60 | 373.98 | 371.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 369.35 | 373.98 | 371.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 371.90 | 373.57 | 371.59 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 363.75 | 369.71 | 370.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 363.35 | 365.68 | 367.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 363.25 | 363.16 | 365.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 363.25 | 363.16 | 365.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 365.70 | 363.68 | 365.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 365.70 | 363.68 | 365.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 365.10 | 363.97 | 365.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 363.95 | 364.11 | 364.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:00:00 | 362.85 | 363.85 | 364.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:00:00 | 363.30 | 363.74 | 364.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 363.45 | 364.17 | 364.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 365.45 | 363.72 | 364.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 371.20 | 363.72 | 364.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 382.30 | 367.44 | 365.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 382.30 | 367.44 | 365.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 388.50 | 377.91 | 375.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 382.30 | 382.82 | 380.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:00:00 | 382.30 | 382.82 | 380.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 381.45 | 382.55 | 380.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 381.45 | 382.55 | 380.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 380.75 | 382.64 | 381.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 378.80 | 382.64 | 381.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 379.60 | 382.03 | 381.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 380.25 | 382.03 | 381.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 379.65 | 381.55 | 380.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:45:00 | 379.25 | 381.55 | 380.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 380.00 | 380.70 | 380.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:15:00 | 380.45 | 380.70 | 380.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 380.45 | 380.65 | 380.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 383.95 | 380.65 | 380.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 383.05 | 385.16 | 385.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 383.05 | 385.16 | 385.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 382.65 | 384.33 | 384.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 380.40 | 380.10 | 381.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 380.40 | 380.10 | 381.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 380.40 | 380.10 | 381.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 377.80 | 379.67 | 380.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 376.05 | 379.36 | 380.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 358.91 | 362.64 | 365.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 357.25 | 362.00 | 364.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 365.00 | 362.60 | 364.87 | SL hit (close>ema200) qty=0.50 sl=362.60 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 371.90 | 366.55 | 365.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 373.50 | 367.94 | 366.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 373.00 | 374.90 | 371.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 373.00 | 374.90 | 371.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 373.00 | 374.90 | 371.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 376.00 | 374.53 | 372.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:15:00 | 375.05 | 374.53 | 372.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:00:00 | 377.45 | 379.61 | 378.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-07 09:15:00 | 413.60 | 401.50 | 394.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 388.85 | 396.71 | 396.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 12:15:00 | 385.45 | 391.15 | 393.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 375.25 | 374.63 | 378.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 11:15:00 | 373.50 | 374.55 | 378.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 15:15:00 | 371.70 | 373.90 | 376.85 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:30:00 | 372.40 | 373.68 | 376.21 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:30:00 | 372.70 | 373.39 | 375.85 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 374.80 | 373.81 | 374.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 370.70 | 372.37 | 373.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 369.65 | 372.33 | 373.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 370.95 | 369.05 | 369.99 | SL hit (close>ema400) qty=1.00 sl=369.99 alert=retest1 |

### Cycle 15 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 377.20 | 370.90 | 370.50 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 370.50 | 371.88 | 371.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 366.65 | 370.40 | 371.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 11:15:00 | 368.75 | 367.94 | 369.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 11:15:00 | 368.75 | 367.94 | 369.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 368.75 | 367.94 | 369.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 368.75 | 367.94 | 369.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 368.50 | 368.05 | 369.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:15:00 | 370.50 | 368.05 | 369.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 370.80 | 368.60 | 369.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:15:00 | 371.15 | 368.60 | 369.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 371.50 | 369.18 | 369.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 372.65 | 369.18 | 369.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 372.50 | 370.05 | 369.77 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 368.35 | 369.98 | 370.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 366.40 | 368.34 | 369.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 367.00 | 366.91 | 367.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:30:00 | 367.35 | 366.91 | 367.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 367.20 | 366.97 | 367.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 367.20 | 366.97 | 367.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 364.30 | 366.44 | 367.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 363.95 | 366.01 | 367.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 363.25 | 365.24 | 366.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 368.20 | 364.02 | 364.95 | SL hit (close>static) qty=1.00 sl=367.70 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 366.55 | 365.55 | 365.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 368.55 | 366.81 | 366.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 367.95 | 368.98 | 367.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 367.95 | 368.98 | 367.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 367.95 | 368.98 | 367.98 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 367.20 | 368.15 | 368.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 366.05 | 367.29 | 367.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 367.95 | 366.94 | 367.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 367.95 | 366.94 | 367.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 367.95 | 366.94 | 367.39 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 369.95 | 367.95 | 367.75 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 366.85 | 368.01 | 368.04 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 368.70 | 368.15 | 368.10 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 366.80 | 367.94 | 368.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 362.00 | 365.71 | 366.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 358.90 | 358.89 | 361.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:45:00 | 360.00 | 358.89 | 361.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 359.05 | 358.90 | 361.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 359.70 | 358.90 | 361.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 360.50 | 359.22 | 361.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 356.00 | 359.22 | 361.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 338.20 | 347.23 | 352.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 13:15:00 | 320.40 | 325.75 | 332.23 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 333.15 | 330.60 | 330.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 339.50 | 332.74 | 331.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 348.45 | 348.61 | 343.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 348.45 | 348.61 | 343.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 342.85 | 346.72 | 343.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 342.85 | 346.72 | 343.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 339.65 | 345.30 | 343.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 339.65 | 345.30 | 343.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 338.00 | 343.84 | 342.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 338.75 | 343.84 | 342.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 337.05 | 341.34 | 341.88 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 345.55 | 342.04 | 341.91 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 340.05 | 341.64 | 341.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 338.80 | 341.07 | 341.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 285.00 | 284.21 | 293.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 295.90 | 286.31 | 292.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 295.90 | 286.31 | 292.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 295.90 | 286.31 | 292.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 308.45 | 290.74 | 294.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 308.45 | 290.74 | 294.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 323.35 | 301.50 | 298.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 335.90 | 313.94 | 305.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 321.80 | 322.80 | 314.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 329.70 | 322.80 | 314.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 322.45 | 324.22 | 322.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 322.45 | 324.22 | 322.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 321.20 | 323.61 | 321.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 320.75 | 323.61 | 321.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 321.65 | 323.22 | 321.93 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 317.70 | 320.79 | 321.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 313.70 | 319.37 | 320.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 309.15 | 308.99 | 311.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 309.15 | 308.99 | 311.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 322.20 | 311.55 | 312.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 324.65 | 311.55 | 312.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 320.20 | 313.28 | 312.91 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 313.20 | 314.38 | 314.48 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 320.20 | 315.54 | 315.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 15:15:00 | 323.00 | 319.28 | 317.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 320.45 | 320.56 | 318.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 12:00:00 | 320.45 | 320.56 | 318.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 317.35 | 319.78 | 318.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 314.45 | 319.78 | 318.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 313.85 | 318.60 | 318.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:15:00 | 313.40 | 318.60 | 318.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 313.85 | 317.65 | 318.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 311.60 | 314.78 | 316.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 315.10 | 313.84 | 315.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 315.10 | 313.84 | 315.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 315.10 | 313.84 | 315.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 315.45 | 313.84 | 315.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 313.30 | 313.73 | 314.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 313.70 | 313.73 | 314.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 313.80 | 313.75 | 314.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 309.80 | 313.62 | 314.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 330.70 | 314.71 | 314.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 330.70 | 314.71 | 314.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 337.50 | 322.46 | 318.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 329.25 | 329.82 | 324.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:00:00 | 329.25 | 329.82 | 324.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 323.75 | 327.75 | 324.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 323.75 | 327.75 | 324.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 323.25 | 326.85 | 324.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 323.25 | 326.85 | 324.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 322.10 | 325.90 | 324.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 320.25 | 324.92 | 323.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 329.00 | 325.73 | 324.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 331.20 | 325.73 | 324.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 14:15:00 | 324.90 | 325.77 | 325.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 324.90 | 325.77 | 325.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 319.40 | 324.42 | 325.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 14:15:00 | 324.75 | 322.68 | 323.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 324.75 | 322.68 | 323.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 324.75 | 322.68 | 323.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:45:00 | 324.15 | 322.68 | 323.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 324.80 | 323.11 | 323.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 324.00 | 323.11 | 323.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 322.25 | 322.93 | 323.76 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 326.55 | 324.37 | 324.14 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 321.35 | 324.10 | 324.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 318.90 | 322.47 | 323.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 316.45 | 314.41 | 316.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 316.45 | 314.41 | 316.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 316.45 | 314.41 | 316.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 316.45 | 314.41 | 316.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 314.50 | 314.43 | 316.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 315.15 | 314.43 | 316.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 314.00 | 313.87 | 315.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:15:00 | 313.00 | 313.87 | 315.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 317.30 | 314.42 | 315.22 | SL hit (close>static) qty=1.00 sl=315.85 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 320.45 | 316.65 | 316.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 331.65 | 321.81 | 319.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 327.50 | 327.79 | 324.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:30:00 | 326.60 | 327.79 | 324.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 325.45 | 326.38 | 324.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 326.25 | 326.38 | 324.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 326.35 | 327.50 | 326.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 322.50 | 325.45 | 325.45 | SL hit (close<static) qty=1.00 sl=323.15 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 322.05 | 324.77 | 325.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 320.60 | 323.94 | 324.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 326.00 | 323.71 | 324.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 326.00 | 323.71 | 324.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 326.00 | 323.71 | 324.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 326.00 | 323.71 | 324.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 326.35 | 324.24 | 324.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 324.90 | 324.24 | 324.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 337.35 | 326.86 | 325.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 337.35 | 326.86 | 325.65 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 324.60 | 329.43 | 329.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 321.30 | 327.81 | 328.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 319.50 | 317.39 | 321.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 319.50 | 317.39 | 321.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 320.15 | 318.34 | 320.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 320.90 | 318.34 | 320.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 322.50 | 319.17 | 321.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 319.80 | 319.32 | 320.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 320.05 | 319.35 | 320.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 320.20 | 319.10 | 319.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 320.60 | 320.16 | 320.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 320.60 | 320.16 | 320.11 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 317.30 | 319.64 | 319.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 12:15:00 | 314.95 | 318.25 | 319.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 317.65 | 317.34 | 318.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 15:15:00 | 317.65 | 317.34 | 318.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 317.65 | 317.34 | 318.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 315.80 | 317.34 | 318.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 315.50 | 314.42 | 314.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 334.50 | 318.61 | 316.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 334.50 | 318.61 | 316.65 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 322.50 | 323.78 | 323.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 320.25 | 322.57 | 323.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 318.85 | 316.66 | 318.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 318.85 | 316.66 | 318.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 318.85 | 316.66 | 318.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 318.85 | 316.66 | 318.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 316.55 | 316.64 | 318.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 320.00 | 316.64 | 318.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 317.20 | 316.76 | 318.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 318.25 | 316.76 | 318.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 318.85 | 317.18 | 318.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 319.25 | 317.18 | 318.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 318.65 | 317.47 | 318.22 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 320.85 | 318.85 | 318.73 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 317.00 | 318.58 | 318.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 316.60 | 318.04 | 318.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 316.70 | 316.49 | 317.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 316.70 | 316.49 | 317.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 316.70 | 316.49 | 317.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 317.05 | 316.49 | 317.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 315.30 | 316.34 | 317.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 314.05 | 315.98 | 316.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 314.10 | 315.17 | 316.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 313.25 | 314.79 | 315.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 313.00 | 313.80 | 314.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 312.15 | 313.47 | 314.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 311.15 | 312.94 | 314.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 298.35 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 298.39 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 297.59 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 297.35 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 295.59 | 302.66 | 306.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 301.70 | 298.41 | 301.42 | SL hit (close>ema200) qty=0.50 sl=298.41 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 309.05 | 302.23 | 301.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 311.75 | 306.34 | 304.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 305.60 | 306.19 | 304.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 305.60 | 306.19 | 304.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 303.50 | 305.65 | 304.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 306.05 | 305.95 | 304.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 302.00 | 304.28 | 304.13 | SL hit (close<static) qty=1.00 sl=302.60 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 302.60 | 303.94 | 303.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 299.75 | 303.10 | 303.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 288.40 | 287.88 | 291.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 293.85 | 287.88 | 291.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 292.45 | 288.79 | 291.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 293.70 | 288.79 | 291.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 289.65 | 288.96 | 291.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 289.30 | 288.96 | 291.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 15:15:00 | 274.83 | 279.52 | 283.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-28 14:15:00 | 260.37 | 266.33 | 271.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 261.85 | 261.32 | 261.32 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 254.45 | 259.95 | 260.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 249.85 | 257.93 | 259.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 254.75 | 254.67 | 257.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 254.75 | 254.67 | 257.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 272.65 | 258.38 | 258.56 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 273.60 | 261.42 | 259.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 280.15 | 264.52 | 261.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 293.40 | 295.09 | 285.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:45:00 | 292.50 | 295.09 | 285.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 294.95 | 293.76 | 289.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:45:00 | 297.50 | 294.24 | 290.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 297.50 | 295.28 | 291.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 299.00 | 295.37 | 292.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 284.80 | 292.80 | 293.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 284.80 | 292.80 | 293.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 10:15:00 | 284.00 | 291.04 | 292.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 273.10 | 272.59 | 276.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:30:00 | 273.00 | 272.59 | 276.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 279.55 | 273.98 | 276.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 279.55 | 273.98 | 276.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 285.15 | 276.22 | 277.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 285.15 | 276.22 | 277.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 285.80 | 279.58 | 278.81 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 13:15:00 | 278.65 | 279.98 | 280.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 277.40 | 278.80 | 279.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 277.15 | 272.67 | 274.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 277.15 | 272.67 | 274.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 277.15 | 272.67 | 274.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 291.40 | 272.67 | 274.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 293.85 | 276.90 | 276.49 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 279.10 | 283.31 | 283.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 278.15 | 280.83 | 281.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 258.20 | 257.80 | 262.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 258.20 | 257.80 | 262.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 259.80 | 258.26 | 261.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 262.90 | 258.26 | 261.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 249.00 | 248.32 | 252.40 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 259.70 | 253.11 | 253.04 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 248.15 | 253.21 | 253.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 246.50 | 251.41 | 252.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 251.95 | 251.24 | 252.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 251.95 | 251.24 | 252.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 251.95 | 251.24 | 252.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 252.75 | 251.24 | 252.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 251.85 | 251.36 | 252.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 250.70 | 251.45 | 252.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 247.85 | 251.48 | 252.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 246.00 | 244.78 | 244.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 246.00 | 244.78 | 244.72 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 241.55 | 244.13 | 244.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 240.75 | 242.56 | 243.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 247.30 | 231.57 | 234.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 247.30 | 231.57 | 234.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 247.30 | 231.57 | 234.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 247.30 | 231.57 | 234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 246.75 | 234.60 | 235.81 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 249.60 | 237.60 | 237.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 254.60 | 241.00 | 238.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 261.00 | 263.08 | 255.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 254.80 | 263.08 | 255.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 256.00 | 261.66 | 255.30 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 242.25 | 251.32 | 252.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 237.80 | 244.29 | 248.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 253.50 | 243.75 | 246.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 253.50 | 243.75 | 246.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 253.50 | 243.75 | 246.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 253.50 | 243.75 | 246.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 260.96 | 247.19 | 248.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 260.96 | 247.19 | 248.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 260.30 | 249.81 | 249.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 268.50 | 255.43 | 252.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 289.94 | 293.44 | 285.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 289.32 | 293.44 | 285.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 289.10 | 291.76 | 286.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 291.04 | 291.76 | 286.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 292.83 | 291.14 | 286.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 289.50 | 291.02 | 288.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 282.24 | 286.95 | 287.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 282.24 | 286.95 | 287.03 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 291.84 | 287.26 | 286.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 292.70 | 288.35 | 287.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 296.99 | 297.61 | 294.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:30:00 | 298.00 | 297.61 | 294.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 293.05 | 296.48 | 295.33 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 292.50 | 294.80 | 294.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 291.75 | 293.89 | 294.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 295.66 | 294.24 | 294.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 295.66 | 294.24 | 294.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 295.66 | 294.24 | 294.50 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 296.50 | 294.69 | 294.68 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 292.73 | 294.46 | 294.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 290.26 | 293.36 | 294.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 292.82 | 292.04 | 292.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 15:15:00 | 292.82 | 292.04 | 292.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 292.82 | 292.04 | 292.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 286.00 | 292.04 | 292.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 281.85 | 279.08 | 278.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 281.85 | 279.08 | 278.83 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 278.25 | 278.65 | 278.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 276.35 | 277.88 | 278.28 | Break + close below crossover candle low |

### Cycle 73 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 286.75 | 278.87 | 278.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 288.40 | 280.78 | 279.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 289.50 | 290.52 | 287.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 289.50 | 290.52 | 287.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:15:00 | 395.65 | 2025-05-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-22 12:30:00 | 396.20 | 2025-05-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 15:15:00 | 395.00 | 2025-05-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 367.25 | 2025-06-24 10:15:00 | 365.90 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-07-03 13:00:00 | 363.95 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.04% |
| SELL | retest2 | 2025-07-03 14:00:00 | 362.85 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-07-03 15:00:00 | 363.30 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2025-07-04 11:30:00 | 363.45 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest2 | 2025-07-15 09:15:00 | 383.95 | 2025-07-18 11:15:00 | 383.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-07-22 14:15:00 | 377.80 | 2025-07-28 14:15:00 | 358.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 376.05 | 2025-07-28 15:15:00 | 357.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:15:00 | 377.80 | 2025-07-29 09:15:00 | 365.00 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-07-23 09:15:00 | 376.05 | 2025-07-29 09:15:00 | 365.00 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2025-07-31 11:45:00 | 376.00 | 2025-08-07 09:15:00 | 413.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 12:15:00 | 375.05 | 2025-08-07 09:15:00 | 412.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 11:00:00 | 377.45 | 2025-08-11 09:15:00 | 388.85 | STOP_HIT | 1.00 | 3.02% |
| SELL | retest1 | 2025-08-18 11:15:00 | 373.50 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest1 | 2025-08-18 15:15:00 | 371.70 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest1 | 2025-08-19 09:30:00 | 372.40 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest1 | 2025-08-19 10:30:00 | 372.70 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-08-20 14:30:00 | 370.70 | 2025-08-26 09:15:00 | 377.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-08-21 12:15:00 | 369.65 | 2025-08-26 09:15:00 | 377.20 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-09-09 09:15:00 | 363.95 | 2025-09-10 09:15:00 | 368.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-09 09:45:00 | 363.25 | 2025-09-10 09:15:00 | 368.20 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-24 09:15:00 | 356.00 | 2025-09-26 09:15:00 | 338.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 356.00 | 2025-09-30 13:15:00 | 320.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-11 13:00:00 | 309.80 | 2025-11-12 09:15:00 | 330.70 | STOP_HIT | 1.00 | -6.75% |
| BUY | retest2 | 2025-11-14 11:15:00 | 331.20 | 2025-11-17 14:15:00 | 324.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-26 15:15:00 | 313.00 | 2025-11-27 09:15:00 | 317.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-01 14:15:00 | 326.25 | 2025-12-02 15:15:00 | 322.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-02 12:15:00 | 326.35 | 2025-12-02 15:15:00 | 322.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-04 09:15:00 | 324.90 | 2025-12-04 09:15:00 | 337.35 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-12-10 10:45:00 | 319.80 | 2025-12-12 10:15:00 | 320.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-12-10 13:30:00 | 320.05 | 2025-12-12 10:15:00 | 320.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-12-11 14:00:00 | 320.20 | 2025-12-12 10:15:00 | 320.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-12-16 09:15:00 | 315.80 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2025-12-18 15:15:00 | 315.50 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2026-01-06 11:15:00 | 314.05 | 2026-01-09 15:15:00 | 298.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:00:00 | 314.10 | 2026-01-09 15:15:00 | 298.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:00:00 | 313.25 | 2026-01-09 15:15:00 | 297.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 313.00 | 2026-01-09 15:15:00 | 297.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 311.15 | 2026-01-12 09:15:00 | 295.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 314.05 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-01-06 14:00:00 | 314.10 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2026-01-06 15:00:00 | 313.25 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-01-08 09:15:00 | 313.00 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2026-01-08 10:45:00 | 311.15 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-01-14 10:00:00 | 309.35 | 2026-01-14 10:15:00 | 309.05 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2026-01-16 09:45:00 | 306.05 | 2026-01-16 14:15:00 | 302.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-22 11:15:00 | 289.30 | 2026-01-23 15:15:00 | 274.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:15:00 | 289.30 | 2026-01-28 14:15:00 | 260.37 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-09 10:45:00 | 297.50 | 2026-02-11 09:15:00 | 284.80 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-02-09 13:00:00 | 297.50 | 2026-02-11 09:15:00 | 284.80 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-02-10 09:15:00 | 299.00 | 2026-02-11 09:15:00 | 284.80 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2026-03-12 14:15:00 | 250.70 | 2026-03-18 15:15:00 | 246.00 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2026-03-13 09:15:00 | 247.85 | 2026-03-18 15:15:00 | 246.00 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2026-04-10 09:15:00 | 291.04 | 2026-04-13 11:15:00 | 282.24 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-04-10 10:15:00 | 292.83 | 2026-04-13 11:15:00 | 282.24 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2026-04-10 15:15:00 | 289.50 | 2026-04-13 11:15:00 | 282.24 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-23 09:15:00 | 286.00 | 2026-05-05 09:15:00 | 281.85 | STOP_HIT | 1.00 | 1.45% |
