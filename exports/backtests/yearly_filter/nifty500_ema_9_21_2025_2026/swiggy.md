# Swiggy Ltd. (SWIGGY)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 263.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 42 |
| ALERT2 | 43 |
| ALERT2_SKIP | 22 |
| ALERT3 | 109 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 67 |
| PARTIAL | 23 |
| TARGET_HIT | 8 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 34
- **Target hits / Stop hits / Partials:** 8 / 59 / 23
- **Avg / median % per leg:** 2.06% / 3.22%
- **Sum % (uncompounded):** 185.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 8 | 34.8% | 4 | 19 | 0 | 0.35% | 8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 8 | 34.8% | 4 | 19 | 0 | 0.35% | 8.0% |
| SELL (all) | 67 | 48 | 71.6% | 4 | 40 | 23 | 2.64% | 177.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 67 | 48 | 71.6% | 4 | 40 | 23 | 2.64% | 177.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 90 | 56 | 62.2% | 8 | 59 | 23 | 2.06% | 185.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 316.55 | 312.61 | 312.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 321.85 | 315.51 | 313.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 321.90 | 322.81 | 320.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 321.90 | 322.81 | 320.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 317.00 | 321.53 | 320.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 315.90 | 321.53 | 320.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 318.95 | 321.01 | 320.09 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 314.40 | 319.11 | 319.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 312.65 | 317.82 | 318.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 314.30 | 313.57 | 315.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 314.30 | 313.57 | 315.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 315.40 | 314.15 | 315.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 315.75 | 314.15 | 315.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 315.10 | 314.34 | 315.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:30:00 | 316.20 | 314.34 | 315.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 316.70 | 314.81 | 315.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 316.95 | 314.81 | 315.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 315.40 | 314.93 | 315.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:00:00 | 314.20 | 314.78 | 315.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 317.55 | 315.67 | 315.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 317.55 | 315.67 | 315.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 320.20 | 316.57 | 316.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 321.00 | 321.27 | 319.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 12:00:00 | 321.00 | 321.27 | 319.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 320.00 | 320.85 | 319.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 323.65 | 321.24 | 320.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 322.45 | 321.00 | 320.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 325.60 | 321.53 | 321.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 11:15:00 | 356.01 | 342.58 | 338.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 353.05 | 361.54 | 362.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 351.05 | 355.66 | 357.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 354.85 | 354.31 | 356.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 354.85 | 354.31 | 356.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 356.50 | 354.60 | 355.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 357.10 | 354.60 | 355.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 358.80 | 355.44 | 356.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 357.60 | 355.44 | 356.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 357.20 | 355.79 | 356.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 358.50 | 355.79 | 356.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 361.35 | 357.45 | 356.97 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 355.95 | 357.18 | 357.33 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 361.60 | 357.47 | 357.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 364.30 | 358.83 | 357.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 389.25 | 389.54 | 382.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 389.25 | 389.54 | 382.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 399.05 | 402.39 | 398.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 399.05 | 402.39 | 398.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 399.70 | 401.85 | 399.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 400.00 | 401.42 | 399.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 400.35 | 401.48 | 400.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:45:00 | 400.00 | 401.35 | 400.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 394.35 | 399.33 | 399.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 394.35 | 399.33 | 399.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 392.80 | 395.58 | 397.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 386.95 | 386.00 | 390.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:00:00 | 386.95 | 386.00 | 390.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 391.80 | 386.81 | 388.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 394.65 | 386.81 | 388.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 392.20 | 387.89 | 388.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 392.75 | 387.89 | 388.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 384.05 | 387.27 | 388.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:00:00 | 383.00 | 385.15 | 386.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 383.00 | 384.45 | 386.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 385.05 | 382.02 | 381.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 385.05 | 382.02 | 381.66 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 380.25 | 381.72 | 381.89 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 387.00 | 382.78 | 382.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 392.15 | 386.88 | 384.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 388.40 | 390.49 | 387.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 388.40 | 390.49 | 387.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 388.40 | 390.49 | 387.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 388.40 | 390.49 | 387.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 388.25 | 390.04 | 387.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 390.00 | 390.04 | 387.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 388.95 | 389.95 | 388.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 398.00 | 389.36 | 388.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 384.90 | 388.70 | 388.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 384.90 | 388.70 | 388.84 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 390.10 | 388.32 | 388.28 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 383.85 | 387.55 | 387.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 382.25 | 386.49 | 387.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 386.55 | 386.33 | 387.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 386.55 | 386.33 | 387.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 386.55 | 386.33 | 387.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 386.00 | 386.33 | 387.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 389.00 | 386.86 | 387.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 389.00 | 386.86 | 387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 389.80 | 387.45 | 387.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 389.80 | 387.45 | 387.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 397.60 | 389.48 | 388.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 411.50 | 393.88 | 390.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 417.90 | 419.59 | 412.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 12:00:00 | 417.90 | 419.59 | 412.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 416.60 | 419.00 | 413.12 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 409.15 | 412.36 | 412.46 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 413.00 | 412.41 | 412.33 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 408.35 | 411.60 | 411.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 404.30 | 410.14 | 411.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 407.65 | 407.29 | 408.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 407.65 | 407.29 | 408.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 413.50 | 408.54 | 409.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 413.50 | 408.54 | 409.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 413.80 | 409.59 | 409.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 409.85 | 409.59 | 409.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 09:15:00 | 389.36 | 401.12 | 403.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 402.60 | 395.60 | 398.02 | SL hit (close>ema200) qty=0.50 sl=395.60 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 397.40 | 394.47 | 394.30 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 392.05 | 394.70 | 394.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 384.00 | 392.46 | 393.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 390.85 | 390.20 | 392.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 390.85 | 390.20 | 392.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 390.85 | 390.20 | 392.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 390.85 | 390.20 | 392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 394.35 | 391.03 | 392.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 393.55 | 391.03 | 392.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 395.35 | 391.89 | 392.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 395.35 | 391.89 | 392.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 398.30 | 393.17 | 393.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 403.70 | 395.28 | 393.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 394.55 | 396.61 | 395.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 394.55 | 396.61 | 395.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 394.55 | 396.61 | 395.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 394.55 | 396.61 | 395.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 393.00 | 395.89 | 395.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 393.70 | 395.89 | 395.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 394.00 | 395.51 | 395.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 402.10 | 395.51 | 395.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-21 09:15:00 | 442.31 | 419.91 | 412.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 420.55 | 425.14 | 425.34 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 432.50 | 425.48 | 425.33 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 422.45 | 425.74 | 425.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 15:15:00 | 420.00 | 424.59 | 425.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 420.90 | 414.79 | 418.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 420.90 | 414.79 | 418.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 420.90 | 414.79 | 418.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 420.30 | 414.79 | 418.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 419.40 | 415.71 | 418.43 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 429.10 | 421.68 | 420.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 430.40 | 424.32 | 422.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 10:15:00 | 424.20 | 424.30 | 422.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:30:00 | 425.60 | 424.30 | 422.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 423.25 | 424.47 | 422.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 423.25 | 424.47 | 422.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 425.70 | 424.72 | 423.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 423.20 | 424.72 | 423.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 430.20 | 427.80 | 425.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 425.60 | 427.80 | 425.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 425.85 | 427.86 | 426.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 425.10 | 427.86 | 426.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 427.45 | 427.78 | 426.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:15:00 | 424.55 | 427.78 | 426.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 423.00 | 426.82 | 426.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 423.00 | 426.82 | 426.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 423.45 | 426.15 | 426.17 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 428.15 | 426.55 | 426.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 432.95 | 427.83 | 426.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 437.20 | 442.27 | 438.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 437.20 | 442.27 | 438.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 437.20 | 442.27 | 438.31 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 09:15:00 | 433.45 | 437.00 | 437.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 10:15:00 | 428.35 | 435.27 | 436.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 428.40 | 428.15 | 431.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 428.40 | 428.15 | 431.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 428.40 | 428.15 | 431.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 424.70 | 427.78 | 430.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 424.75 | 427.25 | 430.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 425.10 | 427.25 | 430.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:30:00 | 425.10 | 426.15 | 428.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 424.70 | 423.98 | 425.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 425.85 | 423.98 | 425.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 425.20 | 424.22 | 425.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 424.60 | 424.22 | 425.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 423.70 | 424.12 | 425.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 425.40 | 424.12 | 425.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 426.50 | 424.59 | 425.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 429.30 | 424.59 | 425.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 430.50 | 425.78 | 425.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 431.15 | 425.78 | 425.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 434.15 | 427.45 | 426.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 434.15 | 427.45 | 426.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 436.05 | 429.17 | 427.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 448.15 | 451.47 | 445.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:45:00 | 447.95 | 451.47 | 445.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 448.20 | 449.51 | 447.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 448.35 | 449.51 | 447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 448.20 | 449.14 | 447.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 445.70 | 449.14 | 447.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 446.60 | 448.63 | 447.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:45:00 | 446.40 | 448.63 | 447.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 447.30 | 448.36 | 447.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 449.00 | 448.36 | 447.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 440.75 | 447.14 | 447.09 | SL hit (close<static) qty=1.00 sl=446.60 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 444.00 | 446.51 | 446.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 440.05 | 444.27 | 445.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 418.50 | 418.49 | 424.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 418.50 | 418.49 | 424.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 424.75 | 419.41 | 423.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 424.55 | 419.41 | 423.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 421.20 | 419.77 | 422.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 419.45 | 419.52 | 422.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 416.45 | 421.53 | 422.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 418.10 | 417.86 | 419.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 421.75 | 418.26 | 418.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 421.75 | 418.26 | 418.16 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 416.00 | 417.84 | 418.03 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 420.00 | 418.27 | 418.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 421.70 | 419.00 | 418.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 418.45 | 419.35 | 418.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 418.45 | 419.35 | 418.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 418.45 | 419.35 | 418.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 421.50 | 419.78 | 419.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 426.80 | 420.55 | 419.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 430.00 | 441.77 | 442.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 430.00 | 441.77 | 442.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 426.40 | 438.70 | 440.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 431.15 | 425.98 | 430.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 431.15 | 425.98 | 430.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 431.15 | 425.98 | 430.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 430.50 | 425.98 | 430.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 433.25 | 427.43 | 430.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 433.25 | 427.43 | 430.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 427.50 | 427.44 | 430.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:30:00 | 425.40 | 426.32 | 428.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 424.00 | 426.32 | 428.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 424.85 | 426.00 | 428.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 423.70 | 427.00 | 428.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 424.25 | 425.87 | 427.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 422.85 | 425.27 | 426.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 422.40 | 424.47 | 425.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 423.10 | 423.05 | 424.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 421.60 | 418.63 | 420.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 409.45 | 416.79 | 419.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 404.40 | 412.46 | 415.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 404.13 | 411.51 | 415.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 403.61 | 411.51 | 415.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 12:15:00 | 402.80 | 408.29 | 412.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 15:15:00 | 402.51 | 405.55 | 410.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 15:15:00 | 401.71 | 405.55 | 410.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 15:15:00 | 401.94 | 405.55 | 410.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 408.20 | 406.08 | 409.95 | SL hit (close>ema200) qty=0.50 sl=406.08 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 395.25 | 391.76 | 391.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 399.55 | 394.27 | 392.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 13:15:00 | 395.45 | 395.96 | 394.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 395.45 | 395.96 | 394.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 393.05 | 395.38 | 394.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 393.05 | 395.38 | 394.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 393.45 | 394.99 | 394.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 396.80 | 394.99 | 394.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 388.85 | 396.33 | 396.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 388.85 | 396.33 | 396.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 387.15 | 393.23 | 394.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 390.55 | 389.00 | 391.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 09:30:00 | 390.00 | 389.00 | 391.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 389.85 | 389.17 | 391.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 390.60 | 389.17 | 391.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 405.70 | 392.22 | 392.21 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 391.45 | 393.01 | 393.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 388.50 | 391.82 | 392.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 382.90 | 381.51 | 384.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 382.90 | 381.51 | 384.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 386.95 | 382.60 | 385.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 386.90 | 382.60 | 385.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 386.95 | 383.47 | 385.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 387.25 | 383.47 | 385.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 388.80 | 386.40 | 386.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 391.40 | 387.62 | 386.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 14:15:00 | 401.75 | 402.39 | 399.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 15:00:00 | 401.75 | 402.39 | 399.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 399.80 | 401.75 | 399.56 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 394.60 | 398.69 | 398.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 392.30 | 396.70 | 397.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 397.70 | 390.86 | 393.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 397.70 | 390.86 | 393.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 397.70 | 390.86 | 393.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 397.70 | 390.86 | 393.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 392.35 | 391.16 | 393.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 395.20 | 391.16 | 393.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 393.80 | 391.69 | 393.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 393.50 | 391.69 | 393.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 394.95 | 392.34 | 393.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 394.95 | 392.34 | 393.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 395.55 | 392.98 | 393.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 397.30 | 392.98 | 393.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 397.80 | 393.95 | 394.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 397.80 | 393.95 | 394.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 397.70 | 394.70 | 394.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 403.75 | 396.51 | 395.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 393.90 | 398.89 | 397.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 393.90 | 398.89 | 397.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 393.90 | 398.89 | 397.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 393.90 | 398.89 | 397.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 398.00 | 398.71 | 397.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 398.90 | 398.71 | 397.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 402.00 | 408.23 | 408.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 402.00 | 408.23 | 408.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 399.00 | 406.38 | 407.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 403.20 | 402.18 | 404.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 11:15:00 | 405.05 | 402.18 | 404.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 402.75 | 402.29 | 404.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 403.20 | 402.29 | 404.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 404.00 | 402.63 | 404.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 404.65 | 402.63 | 404.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 400.40 | 402.19 | 404.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:45:00 | 398.70 | 401.67 | 403.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 397.75 | 401.67 | 403.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 14:15:00 | 411.80 | 402.74 | 402.98 | SL hit (close>static) qty=1.00 sl=404.75 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 408.90 | 403.97 | 403.52 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 402.00 | 405.26 | 405.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 401.00 | 404.41 | 405.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 405.15 | 403.45 | 404.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 405.15 | 403.45 | 404.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 405.15 | 403.45 | 404.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 405.15 | 403.45 | 404.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 403.55 | 403.47 | 404.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:30:00 | 403.20 | 403.56 | 404.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 403.10 | 403.52 | 404.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 400.70 | 403.52 | 404.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 402.75 | 395.25 | 397.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 397.95 | 395.79 | 397.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 397.00 | 396.17 | 397.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 383.04 | 388.08 | 389.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 382.94 | 388.08 | 389.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 382.61 | 388.08 | 389.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 387.80 | 386.47 | 388.55 | SL hit (close>ema200) qty=0.50 sl=386.47 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 323.35 | 317.39 | 317.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 324.60 | 318.83 | 317.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 15:15:00 | 323.85 | 324.50 | 321.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 308.60 | 324.50 | 321.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 307.80 | 321.16 | 320.49 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 309.75 | 318.88 | 319.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 305.15 | 316.13 | 318.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 15:15:00 | 314.90 | 311.85 | 315.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 314.90 | 311.85 | 315.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 314.90 | 311.85 | 315.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 319.65 | 313.54 | 315.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 323.40 | 315.51 | 316.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 323.40 | 315.51 | 316.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 315.55 | 316.16 | 316.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 318.10 | 316.16 | 316.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 315.00 | 315.92 | 316.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:30:00 | 317.55 | 315.92 | 316.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 307.85 | 313.76 | 315.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 306.35 | 313.76 | 315.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 306.55 | 311.69 | 313.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 13:15:00 | 305.45 | 310.86 | 313.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 317.20 | 310.49 | 312.16 | SL hit (close>static) qty=1.00 sl=315.60 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 320.80 | 313.74 | 313.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 327.10 | 322.00 | 320.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 344.60 | 346.33 | 337.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 344.60 | 346.33 | 337.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 333.25 | 341.79 | 339.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 332.00 | 341.79 | 339.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 329.10 | 339.25 | 338.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 329.10 | 339.25 | 338.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 331.65 | 337.73 | 337.91 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 341.40 | 337.85 | 337.38 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 10:15:00 | 332.35 | 337.63 | 337.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 12:15:00 | 330.90 | 335.43 | 336.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 332.95 | 332.47 | 334.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:45:00 | 332.55 | 332.47 | 334.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 334.30 | 332.84 | 334.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 334.30 | 332.84 | 334.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 334.00 | 333.07 | 334.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 335.30 | 333.07 | 334.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 332.30 | 332.92 | 333.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 328.70 | 332.25 | 333.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 312.26 | 320.10 | 323.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 307.65 | 307.08 | 309.82 | SL hit (close>ema200) qty=0.50 sl=307.08 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 299.00 | 296.70 | 296.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 300.45 | 297.45 | 296.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 293.75 | 298.38 | 297.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 293.75 | 298.38 | 297.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 293.75 | 298.38 | 297.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 11:45:00 | 297.60 | 298.20 | 297.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 299.35 | 299.55 | 298.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 296.30 | 298.05 | 298.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 296.30 | 298.05 | 298.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 14:15:00 | 294.30 | 297.30 | 297.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 09:15:00 | 284.50 | 283.46 | 287.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 284.50 | 283.46 | 287.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 284.50 | 283.46 | 287.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 12:15:00 | 280.80 | 283.31 | 286.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:45:00 | 280.80 | 282.91 | 285.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:00:00 | 280.35 | 282.31 | 284.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 290.05 | 282.90 | 283.41 | SL hit (close>static) qty=1.00 sl=287.75 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 290.90 | 284.50 | 284.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 292.30 | 287.12 | 285.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 287.35 | 294.88 | 292.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 287.35 | 294.88 | 292.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 287.35 | 294.88 | 292.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 287.35 | 294.88 | 292.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 286.90 | 293.28 | 291.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 286.85 | 293.28 | 291.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 283.75 | 289.83 | 290.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 282.70 | 288.41 | 289.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 289.55 | 287.85 | 289.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 289.55 | 287.85 | 289.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 289.55 | 287.85 | 289.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 289.85 | 287.85 | 289.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 285.60 | 287.40 | 288.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 284.80 | 286.89 | 288.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 283.65 | 286.05 | 287.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 270.56 | 277.20 | 281.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 269.47 | 277.20 | 281.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 279.60 | 275.18 | 278.90 | SL hit (close>ema200) qty=0.50 sl=275.18 alert=retest2 |

### Cycle 55 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 275.50 | 267.39 | 267.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 281.80 | 272.07 | 270.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 271.60 | 275.95 | 273.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 271.60 | 275.95 | 273.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 271.60 | 275.95 | 273.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 270.20 | 275.95 | 273.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 275.30 | 275.82 | 274.10 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 272.00 | 273.34 | 273.38 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 275.65 | 273.80 | 273.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 11:15:00 | 280.30 | 275.58 | 274.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 274.50 | 275.86 | 274.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 274.50 | 275.86 | 274.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 274.50 | 275.86 | 274.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 274.50 | 275.86 | 274.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 275.25 | 275.74 | 274.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 276.00 | 275.74 | 274.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 267.00 | 274.03 | 274.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 267.00 | 274.03 | 274.23 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 09:15:00 | 276.60 | 272.14 | 271.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 281.90 | 278.79 | 277.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 278.70 | 279.43 | 278.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 279.40 | 279.43 | 278.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 282.20 | 279.98 | 278.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 284.35 | 280.90 | 279.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 14:15:00 | 286.35 | 287.95 | 288.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 286.35 | 287.95 | 288.16 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 290.75 | 288.40 | 288.32 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 286.50 | 288.02 | 288.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 11:15:00 | 285.45 | 287.51 | 287.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 288.90 | 287.79 | 288.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 288.90 | 287.79 | 288.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 288.90 | 287.79 | 288.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 288.75 | 287.79 | 288.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 288.00 | 287.83 | 288.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 286.50 | 287.60 | 287.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 272.18 | 274.99 | 278.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 15:15:00 | 271.95 | 270.74 | 274.49 | SL hit (close>ema200) qty=0.50 sl=270.74 alert=retest2 |

### Cycle 63 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 278.80 | 275.72 | 275.57 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 274.65 | 275.47 | 275.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 271.80 | 274.74 | 275.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 277.30 | 275.08 | 275.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 277.30 | 275.08 | 275.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 277.30 | 275.08 | 275.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 277.80 | 275.08 | 275.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 277.15 | 275.49 | 275.40 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 274.70 | 275.34 | 275.34 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 277.35 | 275.74 | 275.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 279.60 | 276.51 | 275.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 279.10 | 279.48 | 278.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 279.10 | 279.48 | 278.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 277.85 | 279.21 | 278.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 278.25 | 279.21 | 278.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 279.75 | 279.32 | 278.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:00:00 | 280.85 | 279.63 | 278.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:45:00 | 281.95 | 280.03 | 278.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:15:00 | 281.10 | 280.10 | 278.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 282.80 | 280.20 | 279.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-11 09:15:00 | 269.20 | 278.41 | 278.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-05-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 09:15:00 | 269.20 | 278.41 | 278.51 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 09:15:00 | 301.50 | 2025-05-15 13:15:00 | 316.55 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-05-22 14:00:00 | 314.20 | 2025-05-23 10:15:00 | 317.55 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-27 11:45:00 | 323.65 | 2025-06-04 11:15:00 | 356.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 10:45:00 | 322.45 | 2025-06-04 11:15:00 | 354.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 09:15:00 | 325.60 | 2025-06-04 12:15:00 | 358.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 13:15:00 | 400.00 | 2025-07-01 10:15:00 | 394.35 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-06-30 13:45:00 | 400.35 | 2025-07-01 10:15:00 | 394.35 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-30 14:45:00 | 400.00 | 2025-07-01 10:15:00 | 394.35 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-07 10:00:00 | 383.00 | 2025-07-10 10:15:00 | 385.05 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-07 11:30:00 | 383.00 | 2025-07-10 10:15:00 | 385.05 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-15 12:15:00 | 390.00 | 2025-07-17 10:15:00 | 384.90 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-15 15:00:00 | 388.95 | 2025-07-17 10:15:00 | 384.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-16 09:15:00 | 398.00 | 2025-07-17 10:15:00 | 384.90 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-07-30 09:15:00 | 409.85 | 2025-08-01 09:15:00 | 389.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:15:00 | 409.85 | 2025-08-04 12:15:00 | 402.60 | STOP_HIT | 0.50 | 1.77% |
| BUY | retest2 | 2025-08-13 09:15:00 | 402.10 | 2025-08-21 09:15:00 | 442.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-11 11:45:00 | 424.70 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-09-11 13:30:00 | 424.75 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-11 14:15:00 | 425.10 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-09-12 10:30:00 | 425.10 | 2025-09-16 10:15:00 | 434.15 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-09-23 14:15:00 | 449.00 | 2025-09-24 09:15:00 | 440.75 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-30 11:30:00 | 419.45 | 2025-10-06 14:15:00 | 421.75 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-10-01 09:15:00 | 416.45 | 2025-10-06 14:15:00 | 421.75 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-03 10:45:00 | 418.10 | 2025-10-06 14:15:00 | 421.75 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-08 11:00:00 | 421.50 | 2025-10-17 11:15:00 | 430.00 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2025-10-09 09:15:00 | 426.80 | 2025-10-17 11:15:00 | 430.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-10-23 14:30:00 | 425.40 | 2025-11-03 09:15:00 | 404.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 15:00:00 | 424.00 | 2025-11-03 09:15:00 | 403.61 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-10-24 09:45:00 | 424.85 | 2025-11-03 12:15:00 | 402.80 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-10-24 14:15:00 | 423.70 | 2025-11-03 15:15:00 | 402.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 11:00:00 | 422.85 | 2025-11-03 15:15:00 | 401.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:00:00 | 422.40 | 2025-11-03 15:15:00 | 401.94 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-23 14:30:00 | 425.40 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-10-23 15:00:00 | 424.00 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-10-24 09:45:00 | 424.85 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-10-24 14:15:00 | 423.70 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2025-10-27 11:00:00 | 422.85 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-10-28 12:00:00 | 422.40 | 2025-11-04 09:15:00 | 408.20 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-10-29 09:45:00 | 423.10 | 2025-11-07 09:15:00 | 401.28 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-10-31 09:45:00 | 421.60 | 2025-11-07 09:15:00 | 400.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:45:00 | 423.10 | 2025-11-10 14:15:00 | 380.16 | TARGET_HIT | 0.50 | 10.15% |
| SELL | retest2 | 2025-11-03 09:15:00 | 404.40 | 2025-11-10 14:15:00 | 384.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:15:00 | 406.35 | 2025-11-10 14:15:00 | 386.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 11:15:00 | 408.10 | 2025-11-10 14:15:00 | 387.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 12:30:00 | 407.70 | 2025-11-10 14:15:00 | 387.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:45:00 | 421.60 | 2025-11-11 09:15:00 | 379.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-03 09:15:00 | 404.40 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-11-06 09:15:00 | 406.35 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-11-06 11:15:00 | 408.10 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-11-06 12:30:00 | 407.70 | 2025-11-11 12:15:00 | 393.80 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-11-10 13:00:00 | 395.90 | 2025-11-17 12:15:00 | 395.25 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-11-12 10:30:00 | 396.00 | 2025-11-17 12:15:00 | 395.25 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-11-19 09:15:00 | 396.80 | 2025-11-21 09:15:00 | 388.85 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-11 09:15:00 | 398.90 | 2025-12-16 11:15:00 | 402.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-12-17 14:45:00 | 398.70 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-12-17 15:15:00 | 397.75 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-12-24 09:30:00 | 403.20 | 2026-01-02 11:15:00 | 383.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 10:30:00 | 403.10 | 2026-01-02 11:15:00 | 382.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:15:00 | 400.70 | 2026-01-02 11:15:00 | 382.61 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2025-12-24 09:30:00 | 403.20 | 2026-01-02 14:15:00 | 387.80 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-12-24 10:30:00 | 403.10 | 2026-01-02 14:15:00 | 387.80 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-12-24 11:15:00 | 400.70 | 2026-01-02 14:15:00 | 387.80 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-12-29 09:30:00 | 402.75 | 2026-01-05 09:15:00 | 380.66 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2025-12-29 12:15:00 | 397.00 | 2026-01-05 13:15:00 | 377.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:30:00 | 402.75 | 2026-01-06 09:15:00 | 360.63 | TARGET_HIT | 0.50 | 10.46% |
| SELL | retest2 | 2025-12-29 12:15:00 | 397.00 | 2026-01-06 13:15:00 | 357.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-02 10:15:00 | 306.35 | 2026-02-03 09:15:00 | 317.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2026-02-02 12:15:00 | 306.55 | 2026-02-03 09:15:00 | 317.20 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2026-02-02 13:15:00 | 305.45 | 2026-02-03 09:15:00 | 317.20 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-02-19 10:45:00 | 328.70 | 2026-02-24 09:15:00 | 312.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:45:00 | 328.70 | 2026-02-27 10:15:00 | 307.65 | STOP_HIT | 0.50 | 6.40% |
| BUY | retest2 | 2026-03-09 11:45:00 | 297.60 | 2026-03-10 13:15:00 | 296.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-03-10 10:00:00 | 299.35 | 2026-03-10 13:15:00 | 296.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-13 12:15:00 | 280.80 | 2026-03-17 09:15:00 | 290.05 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-03-13 13:45:00 | 280.80 | 2026-03-17 09:15:00 | 290.05 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-03-16 10:00:00 | 280.35 | 2026-03-17 09:15:00 | 290.05 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2026-03-20 12:45:00 | 284.80 | 2026-03-23 13:15:00 | 270.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 283.65 | 2026-03-23 13:15:00 | 269.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:45:00 | 284.80 | 2026-03-24 11:15:00 | 279.60 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest2 | 2026-03-20 13:30:00 | 283.65 | 2026-03-24 11:15:00 | 279.60 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2026-03-25 10:00:00 | 284.20 | 2026-03-27 14:15:00 | 269.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 10:00:00 | 284.20 | 2026-04-01 09:15:00 | 269.35 | STOP_HIT | 0.50 | 5.23% |
| BUY | retest2 | 2026-04-10 15:15:00 | 276.00 | 2026-04-13 09:15:00 | 267.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-21 10:30:00 | 284.35 | 2026-04-24 14:15:00 | 286.35 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-04-27 14:30:00 | 286.50 | 2026-04-30 09:15:00 | 272.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 286.50 | 2026-04-30 15:15:00 | 271.95 | STOP_HIT | 0.50 | 5.08% |
| BUY | retest2 | 2026-05-08 12:00:00 | 280.85 | 2026-05-11 09:15:00 | 269.20 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2026-05-08 12:45:00 | 281.95 | 2026-05-11 09:15:00 | 269.20 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2026-05-08 14:15:00 | 281.10 | 2026-05-11 09:15:00 | 269.20 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2026-05-08 15:15:00 | 282.80 | 2026-05-11 09:15:00 | 269.20 | STOP_HIT | 1.00 | -4.81% |
